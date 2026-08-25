import os, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union, TextIO
import numpy as np
from tqdm import tqdm
from PIL import Image
import math

import time
from utils import classify_gaze

from metrics import (
    multi_hot_targets,
    auc,
    L2_dist,
    ap,
    acc,
    classwise_metrics,
    dark_inference,
    argmax_pts
)

# ---- helpers ---------------------------------------------------------------
def _csv_writer(predict_file, headers):
    """Return a csv.DictWriter that writes header once."""
    writer = csv.DictWriter(predict_file, fieldnames=headers)
    try:
        if predict_file.tell() == 0:
            writer.writeheader()
    except Exception:
        pass
    return writer

def eval_batch( 
        dataset_name: str,
        data: dict,
        output_dict: dict, 
        results:dict, 
        eval_inout: bool = True,
        eval_pattern: bool = True,
        pattern_type: str = 'multi_class',
        get_pattern_from_rules:bool = False,
        skip_auc_computation:bool = False,
        use_dark_inference:bool = False,
        return_heatmap_output:bool = False, #[FOR VISUALIZATION ONLY]
        return_headbox_output:bool = False, #[FOR VISUALIZATION ONLY]
        ):
    """
    Processes a batch and updates metrics.
    Gaze point and heatmap groundtruth is required in this case
    """
    BN = len(data['path'])

    if eval_pattern and get_pattern_from_rules:
        rule_pattern_info = {}

    for person in output_dict.keys():
        if 'pred_confidence' in results.keys():
            if output_dict[person]['pred_heatmap_conf'] is None: 
                raise ValueError("Confidence not returned by model but required, please check.")
            pred_confidence = (
                output_dict[person]['pred_heatmap_conf']
                .detach().cpu().numpy().reshape(-1)
            )
            results['pred_confidence'].extend(pred_confidence)
        
        if eval_pattern and get_pattern_from_rules:
            rule_pattern_info[person] = {}
            rule_pattern_info[person]['gaze'] = []
            rule_pattern_info[person]['headboxes'] = []

        with_outside_gaze = ('inout' in data[person])

        # Get in-out Information
        if with_outside_gaze and (data[person]['inout'] is not None):
            gt_inouts = data[person]["inout"].cpu().detach().numpy()
            if eval_inout:
                pred_inouts = (
                    output_dict[person]["pred_inouts"]
                    .detach().cpu().numpy().reshape(-1)
                )

        # Get Pattern Information
        if eval_pattern:
            gt_patterns = data[person]['pattern'].cpu().detach().numpy()
            if 'pred_patterns' in output_dict[person] and output_dict[person]['pred_patterns'] is not None:
                pred_patterns = output_dict[person]['pred_patterns'].cpu().detach().numpy()
            
        # Get Gaze Follow Information
        if output_dict[person]['pred_heatmap'] is not None:
            pred_heatmap = output_dict[person]['pred_heatmap'].detach().cpu().numpy()
            if pred_heatmap.ndim == 4 and pred_heatmap.shape[1] == 1:
                pred_heatmap = pred_heatmap[:, 0]
            if pred_heatmap.ndim != 3:
                raise ValueError(
                    "Expected batched heatmaps with shape [B, H, W], got "
                    f"{pred_heatmap.shape}."
                )

        for b_i in range(BN):
            gaze_x = float('nan')
            gaze_y = float('nan')
            auc_score = float('nan')
            min_dist = float('nan')
            avg_distance = float('nan')
            norm_pred_x = float('nan')
            norm_pred_y = float('nan')
            if output_dict[person]['pred_heatmap'] is not None:
                # Get predictions
                pred_x, pred_y = dark_inference(pred_heatmap[b_i]) if use_dark_inference else argmax_pts(pred_heatmap[b_i])
                norm_p = [
                    pred_x / pred_heatmap[b_i].shape[-2],
                    pred_y / pred_heatmap[b_i].shape[-1],
                ]
                norm_pred_x = norm_p[0]
                norm_pred_y = norm_p[1]
                # Get Gaze Groundtruth            
                if "gaze" in data[person].keys():
                    if dataset_name =='gazefollow':
                        gazes = data[person]["gaze"][b_i]
                    else:
                        gazes = data[person]["gaze"][b_i].unsqueeze(0)
                    valid_gaze = gazes
                    valid_gaze = valid_gaze[valid_gaze != -1].view(-1, 2)
                    gt_mean = torch.mean(valid_gaze, 0)
                    gaze_x = float(gt_mean[0].item())
                    gaze_y = float(gt_mean[1].item())
                    multi_hot = multi_hot_targets(gazes, data["imsize"][b_i])
                    scaled_heatmap = np.array(Image.fromarray(pred_heatmap[b_i]).resize(data['imsize'][b_i].numpy()))

                    if((not with_outside_gaze) or gt_inouts[b_i]):
                        # Calculate AUC and distance
                        try:
                            if skip_auc_computation:
                                auc_score = float('nan')
                            else:
                                auc_score = auc(multi_hot, scaled_heatmap, is_im=True)
                        # DEBUG ONLY
                        except ValueError as e:
                            print(f"Shape of scaled_heatmap: {scaled_heatmap.shape}")
                            print(f"Contains NaN in scaled_heatmap: {np.isnan(scaled_heatmap).any()}")
                            print(f"Shape of multi_hot: {multi_hot.shape}")
                            print(f"Contains NaN in multi_hot: {np.isnan(multi_hot).any()}")
                            print(f"Error calculating AUC for batch index {b_i}: {e}")

                        all_distances = []
                        for gt_gaze in valid_gaze:
                            all_distances.append(L2_dist(gt_gaze, norm_p))
                        
                        min_dist = float(min(all_distances)) if len(all_distances) else float('nan')
                        
                        if dataset_name=='gazefollow':
                            mean_gt_gaze = torch.mean(valid_gaze, 0)
                            avg_distance = float(L2_dist(mean_gt_gaze, norm_p))
            
            # --- Append to results (per-sample row) ---
            results['image_path'].append(data["path"][b_i])
            results['personID'].append(person)
            results['gaze_x'].append(gaze_x)
            results['gaze_y'].append(gaze_y)
            results['pred_x'].append(float(norm_pred_x))
            results['pred_y'].append(float(norm_pred_y))
            results['gt_inframe'].append(gt_inouts[b_i].item() if eval_inout else "" )
            results['pred_inframe'].append(pred_inouts[b_i].item() if eval_inout else "")
            results['gt_pattern'].append(gt_patterns[b_i].item() if eval_pattern else "")
            if (not get_pattern_from_rules):
                results['pred_pattern'].append(pred_patterns[b_i].item() if eval_pattern else "")

            results['l2_dist'].append(min_dist)
            results['AUC'].append(float(auc_score))
            if dataset_name=='gazefollow':
                results['avg_dist'].append(avg_distance)

            if eval_pattern and get_pattern_from_rules:
                rule_pattern_info[person]['gaze'].append(np.array(norm_p))
                rule_pattern_info[person]['headboxes'].append(data[person]['head_box'][b_i].cpu().detach().numpy())

            if return_heatmap_output:
                results['pred_heatmap'].append(scaled_heatmap)

            if return_headbox_output:
                results['headboxes'].append(data[person]['head_box'][b_i].cpu().detach().numpy())
                
    if eval_pattern and get_pattern_from_rules:
        persons = list(rule_pattern_info.keys())
        bn = len(rule_pattern_info[persons[0]]['gaze'])
        for i in range(len(persons)):
            for b_i in range(bn):
                main_gaze = rule_pattern_info[persons[i]]['gaze'][b_i]
                main_head = rule_pattern_info[persons[i]]['headboxes'][b_i]

                other_gaze = rule_pattern_info[persons[i^1]]['gaze'][b_i]
                other_head = rule_pattern_info[persons[i^1]]['headboxes'][b_i]

                pred_pattern = classify_gaze(main_gaze, other_gaze, main_head, other_head, pattern_type = pattern_type)
                results['pred_pattern'].append(pred_pattern)

    return results
    
class Evaluator:
    def __init__(self, 
                 model:nn.Module, 
                 device: Union[torch.device, str] = "cuda",
                 pattern_type: str = 'multi_class'):
        
        self.model = model
        self.device = device
        self.enable_logging = True
        if pattern_type not in ["multi_class", 
                                "binary_share", 
                                'binary_mutual', 
                                'binary_LAH']:
            raise ValueError("pattern_type must be one of the following \n 'multi_class'/'binary_share'/'binary_mutual'/'binary_LAH'.")
        self.pattern_type = pattern_type
        
    def load_data(self,
        dataset_name: str,
        val_loader: DataLoader,
        eval_pattern: bool = True,
        eval_inout: bool = True,
    ):
        self.dataset_name = dataset_name
        self.val_loader = val_loader
        self.eval_pattern = eval_pattern
        self.eval_inout = eval_inout

    def evaluate(self, 
                model_id: str,
                metric_save_path: Union[str, None] = None,
                pred_save_path: Union[str, None] = None,
                print_classwise_metrics: bool = False,
                get_pattern_from_rules: bool = False,
                get_pred_confidence: bool = False,
                skip_auc_computation: bool = False):
        
        # sanity check that model and dataloader alread loaded
        if self.val_loader is None:
            raise ValueError("No data loaded — call `load_data` first.")

        self.model.eval()

        results = {
            'image_path': [], 'personID': [],
            'gaze_x': [], 'gaze_y': [],
            'pred_x': [], 'pred_y': [],
            'gt_inframe': [], 'pred_inframe': [],
            'gt_pattern': [], 'pred_pattern': []
        }
        if get_pred_confidence:
            results['pred_confidence'] = []
        results['l2_dist'] = []
        results['AUC'] = []

        if self.dataset_name =='gazefollow':
            results['avg_dist'] = []

        predict_file=None
        writer = None
        if pred_save_path:
            predict_file = open(pred_save_path, 'w')
            # Write CSV headers
            headers = list(results.keys())
            writer = _csv_writer(predict_file, headers)

        # Batch evaluation loop
        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(self.val_loader, desc="Inference")):
                output_dict = self.model(data)
                # print(results)
                results = eval_batch(
                                    dataset_name = self.dataset_name,
                                    data=data, 
                                    output_dict=output_dict, 
                                    results=results, 
                                    eval_inout=self.eval_inout,
                                    eval_pattern=self.eval_pattern,
                                    pattern_type=self.pattern_type,
                                    get_pattern_from_rules=get_pattern_from_rules,
                                    skip_auc_computation=skip_auc_computation
                                    )

        # Save Prediction Results
        if writer:
            total_rows = len(next(iter(results.values())))  # get total number of rows
            for i in range(total_rows):
                row = {}
                for key in results:
                    val = results[key][i]
                    # format only real (float-like) numbers, not bools/ints/strings
                    if isinstance(val, (float, np.floating)):
                        row[key] = f"{val:.3f}"  # NaN -> 'nan'
                    else:
                        row[key] = val
                writer.writerow(row)
    
        # Final metrics
        metrics = self.calculate_metrics(results, 
                                         skip_auc_computation)

        if predict_file:
            predict_file.flush()
            try:
                os.fsync(predict_file.fileno())
            except Exception:
                pass
            predict_file.close()


        if self.enable_logging:
            self.log_metrics(metrics, metric_save_path, model_id)

        if print_classwise_metrics:
            if not (self.pattern_type) == "multi_class":
                raise ValueError("Unsupported pattern type of print classification report, \n Class-wise report is available only for multi-class social gaze prediction task.")
            detailed_report = classwise_metrics(results['gt_pattern'], results['pred_pattern'])
            print(detailed_report)
        return metrics
        
    def calculate_metrics(self, results, 
                          skip_auc_computation=False):
        """
        Aggregates and calculates the final metrics.
        """
        metrics = {}

        if skip_auc_computation:
            metrics['AUC']=float('nan')
        else:
            auc_vals = np.asarray(results['AUC'], dtype=np.float64)
            metrics['AUC']=np.nanmean(auc_vals)

        l2_dist_vals = np.asarray(results['l2_dist'], dtype=np.float64)
        metrics['l2_dist']= np.nanmean(l2_dist_vals) 
        
        if 'avg_dist' in results:
            metrics['avg_dist'] = np.nanmean(np.asarray(results['avg_dist'], dtype=np.float64))

        if self.eval_pattern & ('gt_pattern' in results and 'pred_pattern' in results):
            if self.pattern_type == "multi_class":
                metrics['pattern_Acc'] = acc(results['gt_pattern'], results['pred_pattern']) if results['gt_pattern'] else float('nan')
            else:
                metrics['pattern_AP'] = ap(results['gt_pattern'], results['pred_pattern']) if results['gt_pattern'] else float('nan')                
                metrics['pattern_AUC'] = auc(np.array(results['gt_pattern']).astype(int), results['pred_pattern']) if results['gt_pattern'] else float('nan')

        metrics['inframe_AP'] = ap(results['gt_inframe'], results['pred_inframe']) if self.eval_inout else float('nan')
        return metrics

    def log_metrics(self, metrics, metric_save_path, model_id):
        """
        Logs and saves the metrics.
        """
        msg = f"\nEvaluation Results - ModelID {model_id}\n"
        msg += "|AUC    |dist    |Inout   |"

        if 'avg_dist' in metrics:
            msg += "Avg_dist |"
        if 'angular_error' in metrics:
            msg += "Angular  |"
        if 'pattern_Acc' in metrics:
            msg += "Pattern Acc |"
        elif 'pattern_AP' in metrics:
            msg += "Pattern AP |"
        if 'pattern_AUC' in metrics:
            msg += "Pattern AUC |"

        msg += "\n"
        
        metrics_line = f"|{metrics['AUC']:.4f}|{metrics['l2_dist']:.4f}|{metrics['inframe_AP']:.4f}|"
        if 'avg_dist' in metrics:
            metrics_line += f"{metrics['avg_dist']:.4f}|"
        if 'angular_error' in metrics:
            metrics_line += f"{metrics['angular_error']:.4f}°|"
        if 'pattern_Acc' in metrics:
            metrics_line += f"{metrics['pattern_Acc']:.4f}|"
        elif 'pattern_AP' in metrics:
            metrics_line += f"{metrics['pattern_AP']:.4f}|"
        if 'pattern_AUC' in metrics:
            metrics_line += f"{metrics['pattern_AUC']:.4f}|"

        msg += metrics_line + "\n"

        print(msg)
        if metric_save_path:
            with open(metric_save_path, 'a') as f:
                f.write(msg)

    def log_predicts(self,
                     predict_file: TextIO,
                     image_path: str,
                     person_id: str,
                     pred_gaze,
                     pred_inout,
                     pred_pattern,
                     confidence,
                     ):
        values = [
            image_path,
            person_id,
            f"{pred_gaze[0]:.3f}",
            f"{pred_gaze[1]:.3f}",
            f"{pred_inout:.3f}" if not math.isnan(pred_inout) else "nan",
            f"{pred_pattern:.3f}" if not math.isnan(pred_pattern) else "nan",
            f"{confidence:.3f}"
        ]

        predict_file.write(','.join(values) + '\n')
