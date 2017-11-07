import argparse
import numpy as np
import tensorflow as tf
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.python.platform import gfile
from ActivityEnvironment import ActivityEnvironment
from SensorAgent import SensorAgent
from VisionAgent import VisionAgent
from globals import activity_dict_plain,activity_dict_plain_letters, activity_dict_vuzix
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels



def classification_report_to_pandas(ground_truth,
                                    predictions,
                                    full_path="test_pandas.csv",
                                    activity_dict=None):
    """
    Saves the classification report to csv using the pandas module.
    :param ground_truth: list: the true labels
    :param predictions: list: the predicted labels
    :param full_path: string: the path to the file.csv where results will be saved
    :return: None
    """
    import pandas as pd

    # get unique labels / classes
    # - assuming all labels are in the sample at least once
    labels = unique_labels(ground_truth, predictions)

    # get results
    precision, recall, f_score, support = precision_recall_fscore_support(ground_truth,
                                                                          predictions,
                                                                          labels=labels,
                                                                          average=None)
    # a pandas way:
    results_pd = pd.DataFrame({"class": labels,
                               "precision": precision,
                               "recall": recall,
                               "f_score": f_score,
                               "support": support
                               })

    results_pd['class'] = results_pd['class'].apply(pd.to_numeric)
    results_pd = results_pd.sort_values(by=['class']).reset_index(drop=True)
    results_pd['class'] = results_pd['class'].apply(lambda x: activity_dict[x])
    results_pd = results_pd[['class', 'precision', 'recall', 'f_score', 'support']]
    return results_pd


def confusion_matrix_df(y_true, y_pred, activity_dict):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=activity_dict, columns=activity_dict).round(2)
    return df_cm


def evaluate(args):
    sensor_agent = SensorAgent()
    vision_agent = VisionAgent()
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    env = ActivityEnvironment(dataset=args.dataset,
                              sensor_agent=sensor_agent, vision_agent=vision_agent, alpha=0, env_id=0,
                              time=current_time)

    with tf.Session() as sess:
        model_filename = args.policypb
        with gfile.FastGFile(args.sensorpb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_sns, output_sns = tf.import_graph_def(graph_def, return_elements=['lstm_1_input:0', 'output_node0:0'])

        with gfile.FastGFile(args.visionpb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_img, output_img = tf.import_graph_def(graph_def, return_elements=['input_1:0', 'output_node0:0'])

        env.sensor_agent.input_tf = input_sns
        env.sensor_agent.output_tf = output_sns
        env.vision_agent.input_tf = input_img
        env.vision_agent.output_tf = output_img

        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_x, output_x, hidden_1, hidden_2 = tf.import_graph_def(graph_def,
                                                                        return_elements=['global/Placeholder:0',
                                                                                         'global/Softmax:0',
                                                                                         'global/Placeholder_1:0',
                                                                                         'global/Placeholder_2:0'])
        if args.dttype == "multimodal":
            episode_generator = env.episode_generator(stop_at_full_sweep=True, sort=True)
            generator = env.sample_from_episode(episode_generator)
            activity_dict = activity_dict_plain()
        else:
            episode_generator = env.episode_generator_vuzix(stop_at_full_sweep=True)
            generator = env.sample_from_episode_vuzix(episode_generator)
            activity_dict = activity_dict_vuzix()

        y_arr, y_true_arr, actions, y_label_arr, y_true_label_arr, actions_label = [], [], [], [], [], []

        if not args.evaluate_policy:
            args.policypb = args.visionpb if args.evaluate_model == "vision" else args.sensorpb
        actions_probabilities = np.zeros((20, 2))
        activity_count = np.zeros((20, 1))
        while True:

            try:
                done, cur_img_input, cur_img_label, cur_sns_input, cur_sns_label = next(generator)

                if args.evaluate_policy:

                    last_features = [np.zeros((1, 256)), np.zeros((1, 256))]

                    softmax = sess.run([output_x], {input_x: cur_sns_input[np.newaxis, :, :],
                                                    hidden_1: last_features[0], hidden_2: last_features[1]})

                    actions_probabilities[np.argmax(cur_sns_label)] += np.squeeze(softmax)
                    activity_count[np.argmax(cur_sns_label)] += 1
                    action = np.random.choice([0, 1], p=np.squeeze(softmax))

                else:
                    if args.evaluate_model == "Vision":
                        action = env.CAMERA
                    if args.evaluate_model == "Motion":
                        action = env.SENSOR

                if action == env.SENSOR:
                    pred_sns = env.sensor_agent.predict_from_tf(cur_sns_input, session=sess)
                    y = pred_sns
                    y_true = np.argmax(cur_sns_label)

                    y_label = activity_dict_plain()[y]
                    y_true_label = activity_dict_plain()[y_true]
                    action_label = "Motion Agent" if action == 0 else "Vision Agent"

                    y_arr.append(y)
                    y_label_arr.append(y_label)
                    y_true_arr.append(y_true)
                    y_true_label_arr.append(y_true_label)
                    actions.append(action)
                    actions_label.append(action_label)

                if action == env.CAMERA:
                    if not args.evaluate_policy:
                        for img in cur_img_input:
                            pred_img = env.vision_agent.predict_from_tf(img, session=sess)

                            y = pred_img
                            y_true = np.argmax(cur_img_label)

                            y_label = activity_dict_plain()[y]
                            y_true_label = activity_dict_plain()[y_true]
                            action_label = "Motion Agent" if action == 0 else "Vision Agent"

                            y_arr.append(y)
                            y_label_arr.append(y_label)
                            y_true_arr.append(y_true)
                            y_true_label_arr.append(y_true_label)
                            actions.append(action)
                            actions_label.append(action_label)

                    else:
                        pred_img = env.vision_agent.predict_from_tf(cur_img_input, session=sess)

                        y = pred_img
                        y_true = np.argmax(cur_img_label)

                        y_label = activity_dict_plain()[y]
                        y_true_label = activity_dict_plain()[y_true]
                        action_label = "Motion Agent" if action == 0 else "Vision Agent"

                        y_arr.append(y)
                        y_label_arr.append(y_label)
                        y_true_arr.append(y_true)
                        y_true_label_arr.append(y_true_label)
                        actions.append(action)
                        actions_label.append(action_label)

            except StopIteration:
                actions_prob_dict = dict()

                for ix in range(len(actions_probabilities)):
                    actions_prob_dict[activity_dict[ix]] = actions_probabilities[ix] / activity_count[ix]

                stacked = np.column_stack((y_arr, y_label_arr, y_true_arr, y_true_label_arr, actions, actions_label))
                accuracy = sum(stacked[:, 0] == stacked[:, 2]) / float(len(y_true_arr))
                cur_path = args.policypb.split(os.path.sep)
                camera_usage = sum(stacked[:, 4].astype(int)) / len(stacked[:, 4].astype(int))
                sensor_usage = 1 - camera_usage

                base_file_path = os.path.join(*cur_path[:-1], "{0:.2f}".format(accuracy * 100) + '_'
                                              + str(round(sensor_usage, 2)) + '_' + str(round(camera_usage, 2)) + '_'
                                              + cur_path[-1])

                stats_file_path = base_file_path+"_stats"
                cm_file_path = base_file_path+"_cm"
                npy_file_path = base_file_path+".npy"
                action_prob_path = base_file_path+"_action_probs.npy"

                if not args.evaluate_policy:
                    title = "Agent: {} - Accuracy {}%".format(args.evaluate_model, (np.round(accuracy, 2)*100))
                else:
                    title = "λ = {} - Motion Usage {} - Vision Usage {} - Accuracy {}%"\
                        .format(0, round(sensor_usage, 2), round(camera_usage, 2), (round(accuracy, 2)*100))

                get_stats_from_np(stacked,stats_file_path, cm_file_path,
                                  title=title.format(0, round(sensor_usage, 2), round(camera_usage, 2)),
                                  activity_dict=activity_dict)

                np.save(npy_file_path, stacked)
                np.save(action_prob_path, actions_prob_dict)

                print("Final Accuracy {0:.2f}".format(accuracy * 100))
                print("Camera usage {} - Sensor Usage {}".format(round(camera_usage, 2),
                                                                 round(sensor_usage, 2)))
                break


def get_stats_from_np(np_array, stats_file, cm_file, title, activity_dict):
    y_true = np_array[:, 2].astype(int)
    y_pred = np_array[:, 0].astype(int)
    df_stats = classification_report_to_pandas(y_true, y_pred, activity_dict=activity_dict)

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index=activity_dict, columns=activity_dict_plain_letters()).round(2)

    sns.set(rc={"figure.figsize": (25, 23)})
    sns.set(font_scale=1.6)
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 25
    plt.rcParams["ytick.labelsize"] = 25
    hm = sns.heatmap(df_cm, annot=True, cmap="Blues", cbar=False)
    hm.axes.set_title(title, fontsize=32)
    hm.axes.set_xlabel("Predicted")
    hm.axes.set_ylabel("True")
    hm.set_xticklabels(hm.get_xticklabels(), rotation=360)
    hm.figure.savefig(cm_file+".svg", format='svg', dpi=100, bbox_inches='tight')
    hm.figure.savefig(cm_file + ".eps", format='eps', dpi=100, bbox_inches='tight')

    # acc_per_class = [cm.iloc[i][i]/df_stats.iloc[i]['support'] for i in range(len(cm))]
    # df_stats['accuracy'] = pd.Series(acc_per_class, index=df_stats.index)
    # df_stats = df_stats[['class', 'accuracy', 'precision', 'recall', 'f_score', 'support']]
    pd.DataFrame.to_csv(df_stats, stats_file+".csv")
    pd.DataFrame.to_csv(df_cm, cm_file+".csv")
    return df_stats


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--evaluate_policy", action="store_true")
    a.add_argument("--evaluate_model", default="vision", choices=['Vision', 'Motion'])
    a.add_argument("--dataset", default='multimodal_dataset/video/splits/test', required=True)
    a.add_argument("--dttype", default='multimodal', required=True)
    a.add_argument('--sensorpb', default='models/production/tb_models/multimodal_sns_0_611429-0_71.pb',
                   type=str, help='Protobuff File for the Sensor Network', required=True)

    a.add_argument('--visionpb', default='models/production/tb_models/multimodal_inception_010-0_860811-0_79.pb',
                   type=str, help='Protobuff File for the Vision Network', required=True)

    a.add_argument('--policypb', default='models/production/policies/policy_alpha_01.pb', type=str,
                   help='Protobuff File for the Vision Network')

    a.add_argument('--num_runs', default=20, type=int,
                   help='Number of times to run the evaluation')



    args = a.parse_args()
    for _ in range(args.num_runs):
        evaluate(args)
    # np_array = np.load("models/production/tb_models/78.70_0.0_1.0_multimodal_sns_0_611429-0_71.pb.npy")
    # get_stats_from_np(np_array, "stats", "cm", "Agent: {} - Accuracy {}%".format("Vision", 78.70))
