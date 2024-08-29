from utils import (
    parse_args,
    setup_basics,
    neptune_log,
    set_seeds,
)
from utils.online_logs import (
    update_online_metrics,
    reset_avg_online_metrics,
    get_online_metrics_mult,
    log_avg_online,
    log_final,
    log_examples_selected,
    log_strategy_data,
    log_thresholds,
)
import numpy as np
from metrics import Metric
from handler import handler_LLM
from student import student
from accelerate import Accelerator
from accelerate.logging import get_logger
from task import (
    get_task,
    make_datacollator,
)
import pdb
import copy
import gc

logger = get_logger(__name__)


def main():
    args = parse_args()
    print(args)
    accelerator = Accelerator()
    run = setup_basics(accelerator, logger, args)

    # Pre-Logging
    run["args"] = vars(args)
    set_seeds(args.seed)

    task = get_task(
        accelerator=accelerator,
        args=args,
        model=None,
    )
    if not task.is_classification:
        args.is_classification = False
    args.soft_labels = True if args.soft_labels == 1 else False
    online_dataloader = task.data["online_dataloader"]
    st = student(args, task, run, accelerator)
    print('Progress: Loaded Student')
    budgets = [int(b) for b in args.budget.split(",")]

    wrap = handler_LLM(args, st, task)
    print('Progress: Wrap Handler Loaded')
    metric = Metric(args, soft=args.soft_labels, online=True)
    print("Progress: Metric Loaded")
    # Initialize student model
    # If we put a checkpoint, we load the model and we skip the first $checkpoint steps
    if args.checkpoint != "-1":
        PATH = "checkpoints/" + args.task_name + "/" + str(args.checkpoint) + ".pt"
        if args.n_init == 100 and args.strategy == "MV":
            PATH = (
                "checkpoints/"
                + args.task_name
                + "/"
                + str(args.checkpoint.split("_")[0])
                + "_"
                + str(args.checkpoint.split("_")[1])
                + "_500.pt"
            )
        st.init_checkpoint(PATH)
        wrap = handler_LLM(args, st, task)
        print('Progress: Wrap Handler Loaded')
        wrap.student_vec = []
        if args.strategy == "MV":
            for idx in range(5):
                st_aux = student(args, task, run, accelerator)
                aux_name = int(args.checkpoint.split("_")[2])
                if args.n_init == 100:
                    aux_name = 500
                PATH_AUX = (
                    "checkpoints/"
                    + args.task_name
                    + "/"
                    + str(args.checkpoint.split("_")[0])
                    + "_"
                    + str(args.checkpoint.split("_")[1])
                    + "_"
                    + str(aux_name - 400 + 100 * idx)
                    + ".pt"
                )
                st_aux.init_checkpoint(PATH_AUX)
                wrap.student_vec.append(copy.deepcopy(st_aux.model).cpu())
                del st_aux

    stop_retraining = args.strategy == "EM_raw"
    send_update = False
    all_pred = []
    x_samples_pred = []
    print("Progress: Simulation Starts")
    for step, sample in enumerate(online_dataloader):
        print('Step:', step)
        # if using checkpoints, and we are within the first n_init data-points - save LLM response  
        if args.checkpoint != "-1" and step < args.n_init:
            wrap.save_cache(sample, step)
            if args.strategy == "CS":
                wrap.output = wrap.call_llm(sample)
                wrap.obtain_embed(sample)
                wrap.save_embed()
        else:
            gc.collect()
            decision, pred = wrap.query(sample, step)

            stats = get_online_metrics_mult(
                args,
                metric,
                sample,
                pred,
                decision,
                budgets,
                wrap.performance,
                all_pred,
                x_samples_pred
            )
            neptune_log(
                run=run,
                pref=f"online/",
                stats=stats,
                epoch=step,
            )
            if step == 0 or (args.checkpoint != "-1" and step == args.n_init):
                avg_online = reset_avg_online_metrics(stats)
            avg_online = update_online_metrics(avg_online, stats)
            
            if should_retrain(args, wrap, stop_retraining, step):
                print('Retraining student')
                set_seeds(args.seed)

                cache = wrap.retrieve_cache()
                train_dataloader, eval_dataloader = make_datacollator(
                    args, task.tokenizer, cache
                )
                train_dataloader, eval_dataloader = accelerator.prepare(
                    train_dataloader, eval_dataloader
                )

                if wrap.retrain:
                    st.suffixes.append(str(budgets[len(wrap.budget_models)]) + "-")
                st.train(train_dataloader, eval_dataloader)

                del train_dataloader, eval_dataloader
                if step + 1 and (step + 1) % args.retrain_freq == 0: wrap.update = False

                wrap.reorder_students()
                if args.empty_cache == 1:
                    wrap.reset_buffer()
                if wrap.budget_arr[-1] == 0:
                    stop_retraining = True
                    wrap.delete_cache()
                send_update = True

            if send_update or step == len(online_dataloader) - 1:
                log_avg_online(run, avg_online, step, budgets[-1])
                avg_online = reset_avg_online_metrics(stats)
                send_update = False
                if step == len(online_dataloader) - 1:
                    log_examples_selected(run, wrap.steps)
                    log_final(run)
                    if args.strategy == 'BT':
                        log_strategy_data(run, args.strategy, wrap.BT)
                    elif args.strategy == 'EN':
                        log_strategy_data(run, args.strategy, wrap.EN)
                    elif args.strategy == 'CS':
                        log_strategy_data(run, args.strategy, wrap.CS_similarities)       
                    if args.dynamic_threshold == 1 and len(wrap.thresholds) > 0:
                        log_thresholds(run, wrap.thresholds)

    if run is not None:
        run.stop()

def should_retrain(args, wrap, stop_retraining, step):
    '''
    Determines whether the student model should be retrained.
    '''
    if args.retrain_fixed == 1:
        retrain_clause = (step + 1) % args.retrain_freq == 0
    elif hasattr(wrap, "cache") and "input_ids" in wrap.cache:
        retrain_clause = len(wrap.cache["input_ids"]) % args.retrain_freq == 0
        print('retrain_fixed ii', len(wrap.cache["input_ids"]), retrain_clause)
    else: 
        return False
    return wrap.retrain or (step + 1 and retrain_clause and not stop_retraining)

if __name__ == "__main__":
    main()
