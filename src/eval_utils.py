import os
import numpy as np
from log_utils import log


def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : mean absolute error
    '''
    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean((1.0 / tgt - 1.0 / src) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Args:
        src : numpy
            source array
        tgt : numpy
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs(1.0 / tgt - 1.0 / src))

def evaluate(output_depths,
             ground_truths,
             best_results=None,
             step=0,
             session=None,
             saver=None,
             checkpoint_path=None,
             log_path=None,
             min_evaluate_depth=0.0,
             max_evaluate_depth=100.0):
    '''
    Evaluates depth error w.r.t. ground truth

    Args:
        output_depths : list[numpy]
            list of H x W x 1 depth predictions
        ground_truths : list[numpy]
            list of H x W x 1 ground truths
        best_results : dict
            dictionary of best recorded results
        step : int
            step of the trained model
        session : tf.Session
            tensorflow session
        saver : tf.Saver
            checkpoint saver for best checkpoint
        checkpoint_path : str
            path to save checkpoint
        log_path : str
            path to log results
        min_evaluate_depth : float
            minimum depth value to evaluate
        max_evaluate_depth : float
            maximum depth value to evaluate
    Returns:
        dict : best results
    '''

    assert len(output_depths) == len(ground_truths)

    n_sample = len(ground_truths)
    rmse = np.zeros(n_sample, np.float32)
    mae = np.zeros(n_sample, np.float32)
    imae = np.zeros(n_sample, np.float32)
    irmse = np.zeros(n_sample, np.float32)

    for idx in range(n_sample):
        output_depth = np.squeeze(output_depths[idx, ...])
        ground_truth = np.squeeze(ground_truths[idx][..., 0])
        validity_map = np.squeeze(ground_truths[idx][..., 1])

        # Create mask for evaluation
        validity_mask = np.where(validity_map > 0, 1, 0)
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)
        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        # Apply mask to output depth and ground truth
        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute evaluation metrics
        mae[idx] = mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
        rmse[idx] = root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
        imae[idx] = inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
        irmse[idx] = inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Take mean over all metrics
    mae   = np.mean(mae)
    rmse  = np.mean(rmse)
    imae  = np.mean(imae)
    irmse = np.mean(irmse)

    log('Evaluation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse),
        log_path)

    if best_results is not None:
        # Check whether current results is better than best results
        n_improved = 0

        if np.round(mae, 1) <= np.round(best_results['mae'], 1):
            n_improved = n_improved + 1
        if np.round(rmse, 1) <= np.round(best_results['rmse'], 1):
            n_improved = n_improved + 1
        if np.round(imae, 3) <= np.round(best_results['imae'], 3):
            n_improved = n_improved + 1
        if np.round(irmse, 3) <= np.round(best_results['irmse'], 3):
            n_improved = n_improved + 1

        if n_improved > 2:
            best_results['step'] = step
            best_results['mae'] = mae
            best_results['rmse'] = rmse
            best_results['imae'] = imae
            best_results['irmse'] = irmse

            if session is not None and saver is not None and checkpoint_path is not None:
                saver.save(
                    session,
                    os.path.join(checkpoint_path, 'best', 'model.ckpt'),
                    global_step=step)

        log('Best results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            best_results['step'],
            best_results['mae'],
            best_results['rmse'],
            best_results['imae'],
            best_results['irmse']), log_path)

    return best_results
