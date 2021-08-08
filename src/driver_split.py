import pandas as pd
import os

df = pd.read_csv('driver_imgs_list.csv', index_col=False)


def split_csv(to_val_drivers):
    matched_df = df.loc[df['subject'].isin(to_val_drivers)]
    unmatched_df = df.loc[~df['subject'].isin(to_val_drivers)]
    matched_df.to_csv("matched.csv", index=False, header=["subject", "classname", "img"])
    unmatched_df.to_csv("unmatched.csv", index=False, header=["subject", "classname", "img"])


split_csv(["p002", "p081"])


def convert_csv():
    ## TRAIN
    TRAIN_PATH = "./auc.distracted.driver.dataset_v2/v2_cam1_cam2_ split_by_driver/Camera 1/train/"
    train_classnames = []
    train_imgs = []
    c_list = os.listdir(TRAIN_PATH)
    for i in c_list:
        imgs = os.listdir(TRAIN_PATH + i + "/")
        for j in imgs:
            train_classnames += [i]
            train_imgs += [j]
    train_csv = pd.DataFrame({"classname": train_classnames, "img": train_imgs}).sort_values(by=["classname", "img"])
    train_csv.to_csv("auc_train.csv", index=False)

    ## TEST
    TEST_PATH = "./auc.distracted.driver.dataset_v2/v2_cam1_cam2_ split_by_driver/Camera 1/test/"
    test_classnames = []
    test_imgs = []
    c_list = os.listdir(TEST_PATH)
    for i in c_list:
        imgs = os.listdir(TEST_PATH + i + "/")
        for j in imgs:
            test_classnames += [i]
            test_imgs += [j]
    test_csv = pd.DataFrame({"classname": test_classnames, "img": test_imgs}).sort_values(by=["classname", "img"])
    test_csv.to_csv("auc_test.csv", index=False)


convert_csv()
