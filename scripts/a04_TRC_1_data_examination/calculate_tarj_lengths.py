from create_segments_for_WDE_SBG import*
from utils import Classes
import pandas as pd


def calc_traj_length_on_exp(exp_path, GT=None):
    exp = Classes.SbgExpRawData(exp_path, GT=GT)
    dL = exp.calc_dL(window_size=1)
    L = sum(dL)
    return L


def calc_traj_length_on_list(exp_path_list):
    exp_length_dic = {}
    N = len(exp_path_list)
    i = 1
    for exp_path in exp_path_list:
        exp_length_dic[exp_path] = calc_traj_length_on_exp(exp_path)
        print(str(i / N * 100) + '% completed')
        i += 1
    return exp_length_dic


def calc_traj_length_on_person_list(person_list, dir_to_analyze):
    exp_path_list = get_exp_list(list_of_persons=person_list,
                                 dir_to_analyze=dir_to_analyze)
    N = len(exp_path_list)
    i = 1
    exp_length_dic = {}
    for person in person_list:
        person_path = join(dir_to_analyze, person)
        GT_path = join(person_path, 'ascii-output.txt')
        GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
        exp_list_in_person = get_exp_list(list_of_persons=[person],
                                          dir_to_analyze=dir_to_analyze)
        for exp_path in exp_list_in_person:
            exp = Classes.SbgExpRawData(exp_path, GT=GT)
            if exp.valid:
                dL = exp.calc_dL(window_size=1)
                L = sum(dL)
                exp_length_dic[exp_path] = L
            print(str(round(i / N * 100), 2) + '% completed')
            i += 1
    return exp_length_dic


def dic2arr(dic):
    """input is a dictoinary where all values are scalars"""
    keys = dic.keys()
    arr = np.array([])
    for key in keys:
        arr = np.hstack([arr, dic[key]])
    return arr


def calculate_traj_length_statistics(exp_length_dictionary):
    fig = plt.figure('traj lengths divide to modes and train test ')

    traj_lengths_arr_train_swing = dic2arr(exp_length_dictionary["train_swing"])
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set(title="train_swing"), ax1.grid(True)
    ax1.hist(traj_lengths_arr_train_swing, 100)

    traj_lengths_arr_test_swing = dic2arr(exp_length_dictionary["test_swing"])
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set(title="test_swing"), ax2.grid(True)
    ax2.hist(traj_lengths_arr_test_swing, 100)

    traj_lengths_arr_train_pocket = dic2arr(exp_length_dictionary["train_pocket"])
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set(title="train_pocket"), ax3.grid(True)
    ax3.hist(traj_lengths_arr_train_pocket, 100)

    traj_lengths_arr_test_pocket = dic2arr(exp_length_dictionary["test_pocket"])
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set(title="test_pocket"), ax4.grid(True)
    ax4.hist(traj_lengths_arr_test_pocket, 100)

    traj_lengths_arr_train_text = dic2arr(exp_length_dictionary["train_text"])
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.set(title="train_text"), ax5.grid(True)
    ax5.hist(traj_lengths_arr_train_text, 100)

    traj_lengths_arr_test_text = dic2arr(exp_length_dictionary["test_text"])
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.set(title="test_text"), ax6.grid(True)
    ax6.hist(traj_lengths_arr_test_text, 100)

    plt.savefig(join(outputfolder, 'traj_divided_by_modes_train_test'))

    traj_lengths_arr_unufied = np.hstack([exp_length_dictionary["train_swing"],
                                          exp_length_dictionary["test_swing"],
                                          exp_length_dictionary["train_pocket"],
                                          exp_length_dictionary["test_pocket"],
                                          exp_length_dictionary["train_text"],
                                          exp_length_dictionary["test_text"]])
    plt.figure('traj_lengths_SBG.png')
    plt.hist(traj_lengths_arr_unufied, 100)
    plt.grid(True)
    plt.savefig(join(outputfolder, 'traj_lengths_SBG.png'))

    plt.show()


if __name__ == '__main__':
    data_location = 'magneto'
    WD = os.getcwd()
    outputfolder = join(WD, 'data', 'SBG_traj_lenths')

    list_of_train_persons_swing, list_of_test_persons_swing = get_train_and_test_lists(mode='swing')
    list_of_train_persons_pocket, list_of_test_persons_pocket = get_train_and_test_lists(mode='pocket')
    list_of_train_persons_text, list_of_test_persons_text = get_train_and_test_lists(mode='text')

    dir_to_analyze_swing = get_dir_to_analyze(data_location=data_location, mode='swing')
    dir_to_analyze_pocket = get_dir_to_analyze(data_location=data_location, mode='pocket')
    dir_to_analyze_text = get_dir_to_analyze(data_location=data_location, mode='text')

    print('working on list_of_train_persons_swing')
    exp_length_dic_train_swing = calc_traj_length_on_person_list(list_of_train_persons_swing, dir_to_analyze_swing)
    print('working on list_of_test_persons_swing')
    exp_length_dic_test_swing = calc_traj_length_on_person_list(list_of_test_persons_swing, dir_to_analyze_swing)
    print('working on list_of_train_persons_pocket')
    exp_length_dic_train_pocket = calc_traj_length_on_person_list(list_of_train_persons_pocket, dir_to_analyze_pocket)
    print('working on list_of_test_persons_pocket')
    exp_length_dic_test_pocket = calc_traj_length_on_person_list(list_of_test_persons_pocket, dir_to_analyze_pocket)
    print('working on list_of_train_persons_text')
    exp_length_dic_train_text = calc_traj_length_on_person_list(list_of_train_persons_text, dir_to_analyze_text)
    print('working on list_of_test_persons_text')
    exp_length_dic_test_text = calc_traj_length_on_person_list(list_of_test_persons_text, dir_to_analyze_text)
    # exp_list_train_swing = get_exp_list(list_of_persons=list_of_train_persons_swing,
    #                                     dir_to_analyze=dir_to_analyze_swing)
    # exp_list_test_swing = get_exp_list(list_of_persons=list_of_test_persons_swing,
    #                                    dir_to_analyze=dir_to_analyze_swing)
    # exp_list_train_pocket = get_exp_list(list_of_persons=list_of_train_persons_pocket,
    #                                      dir_to_analyze=dir_to_analyze_pocket)
    # exp_list_test_pocket = get_exp_list(list_of_persons=list_of_test_persons_pocket,
    #                                     dir_to_analyze=dir_to_analyze_pocket)
    # exp_list_train_text = get_exp_list(list_of_persons=list_of_train_persons_text,
    #                                    dir_to_analyze=dir_to_analyze_text)
    # exp_list_test_text = get_exp_list(list_of_persons=list_of_test_persons_text,
    #                                   dir_to_analyze=dir_to_analyze_text)
    #
    #
    # exp_length_dic_train_swing = calc_traj_length_on_list(exp_list_train_swing)
    # exp_length_dic_test_swing = calc_traj_length_on_list(exp_list_test_swing)
    # exp_length_dic_train_pocket = calc_traj_length_on_list(exp_list_train_pocket)
    # exp_length_dic_test_pocket = calc_traj_length_on_list(exp_list_test_pocket)
    # exp_length_dic_train_text = calc_traj_length_on_list(exp_list_train_text)
    # exp_length_dic_test_text = calc_traj_length_on_list(exp_list_test_text)




    exp_length_dictionary = {"train_swing": exp_length_dic_train_swing,
                             "test_swing": exp_length_dic_test_swing,
                             "train_pocket": exp_length_dic_train_pocket,
                             "test_pocket": exp_length_dic_test_pocket,
                             "train_text": exp_length_dic_train_text,
                             "test_text": exp_length_dic_test_text}
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    with open(join(outputfolder, 'trajectory_lengths.json'), 'w') as f:
        json.dump(exp_length_dictionary, f, indent=4)
        f.close()

    calculate_traj_length_statistics(exp_length_dictionary)