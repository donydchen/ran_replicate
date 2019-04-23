import os
import argparse
import numpy as np 


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv_path', required=True, help="Path to the result files.")
    parser.add_argument('--aus', type=str, default='1,2,4,5,6,7,9,12,17,23,25', help='AUs vector index.')
    opt = parser.parse_args()

    aus_list = list(map(lambda x: "AU%02d" % int(x), list(opt.aus.split(','))))

    prob_list = []
    with open(opt.csv_path, 'r') as f:
        for line in f.readlines():
            tmp_list = line.strip().split(",")
            if len(tmp_list) < 5:
                continue
            cur_prob = list(map(lambda x: float(x), tmp_list[1:]))
            prob_list.append(np.array(cur_prob))
    prob_arr = np.array(prob_list)

    prob_avg = np.mean(prob_arr, axis=0)
    with open(opt.csv_path, 'a+') as f:
        f.write("Average\n")
        for k, v in zip(aus_list, prob_avg):
            print("%s: %f" % (k, v))
            f.write("%s, %f\n" % (k, v))
        print("Avg : %f" % np.mean(prob_arr))
        f.write("Avg , %f" % np.mean(prob_arr))


if __name__ == "__main__":
    main()
