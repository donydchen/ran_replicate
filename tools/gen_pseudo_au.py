import os
import random
import argparse


class GenPseudoAU(object):
    """docstring for GenPseudoAU"""
    def __init__(self):
        super(GenPseudoAU, self).__init__()
        # init domain knowledge table
        self.init_table()

    def init_table(self):
        self.EXPRESSION = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
        # table is constructed in a look-behind way
        # table 1
        self.EXP_DEP_TABLE = {self.EXPRESSION[0]:{4:1., 7:1., 24:1., 10:0.26, 17:0.52, 23:0.29},
                              self.EXPRESSION[1]:{9:1., 10:1., 17:1., 4:0.31, 24:0.26},
                              self.EXPRESSION[2]:{1:1., 4:1., 20:1., 25:1., 2:0.57, 5:0.63, 26:0.33},
                              self.EXPRESSION[3]:{12:1., 25:1., 6:0.51},
                              self.EXPRESSION[4]:{4:1., 15:1., 1:0.6, 6:0.5, 11:0.26, 17:0.67},
                              self.EXPRESSION[5]:{1:1., 2:1., 25:1., 26:1., 5:0.66}}
        # table 2
        self.EXP_COM_TABLE = {self.EXPRESSION[0]:{5:[[4]], 7:[[4]], 24:[[17]]},
                              self.EXPRESSION[2]:{4:[[1, 2]]},
                              self.EXPRESSION[1]:{},
                              self.EXPRESSION[3]:{12:[[6], [7]]},
                              self.EXPRESSION[4]:{4:[[1]], 15:[[6], [11]], 17:[[11]]},
                              self.EXPRESSION[5]:{5:[[1, 2]], 26:[[1, 2]]}}
        # table 3 
        self.EXP_IND_COE_TABLE = {2:[1], 5:[1, 2], 
                                  7:[4], 9:[4, 7], 
                                  17:[15], 24:[15, 17],
                                  24:[23]}
        self.EXP_IND_MUT_TABLE = {15:[12], 17:[12],
                                  6:[2], 7:[2], 9:[2],
                                  25:[15, 17, 23, 24]}

    def initialize(self, opt):
        # setup option 
        self.aus_list = sorted(map(lambda x: int(x), list(opt.aus.split(','))))
        self.n_sample = int(opt.n_sample)
        self.saved_path = opt.saved_path

    def run(self):
        saved_log = []
        # run the Algorithm 1 
        for expression in self.EXPRESSION:
            print(">>> Start generating for %s..." % expression)
            for idx in range(self.n_sample):
                cur_log = "%s,%s" % (expression, self.gen_aus(expression))
                saved_log.append(cur_log)

        # save result to files
        print("Saving result to %s..." % self.saved_path)
        self.write_to_file(saved_log)

        print("[End] Successfully generate (%dx%d = %d) pseudo AUs." % (self.n_sample, len(self.EXPRESSION), \
                self.n_sample * len(self.EXPRESSION)))

    def gen_aus(self, cur_express):
        # generate one aus 
        tab_1 = self.EXP_DEP_TABLE[cur_express]
        tab_2 = self.EXP_COM_TABLE[cur_express]
        pseudo_aus = {}
        # sample the first AU according to Table 1
        pseudo_aus[self.aus_list[0]] = self.gen_by_table_one(self.aus_list[0], tab_1, pick_num=random.uniform(0., 1.))
        for au_idx in self.aus_list[1:]:
            pick_num = random.uniform(0., 1.)
            if self.meet_table_two(au_idx, tab_2, pseudo_aus):
                pseudo_aus[au_idx] = int(pick_num < random.uniform(0.7, 1.0)) # sample probability in [0.7, 1]

            elif self.meet_table_three(au_idx, self.EXP_IND_COE_TABLE, pseudo_aus):
                pseudo_aus[au_idx] = int(pick_num < random.uniform(0.7, 1.0)) # sample probability in [0.7, 1]

            elif self.meet_table_three(au_idx, self.EXP_IND_MUT_TABLE, pseudo_aus):
                pseudo_aus[au_idx] = int(pick_num < random.uniform(0., 0.2)) # sample probability in [0., 0.2]

            else:
                pseudo_aus[au_idx] = self.gen_by_table_one(au_idx, tab_1, pick_num) # sample according to Table 1 

        # convert dict to list str
        pseudo_aus_list = []
        for au_idx in self.aus_list:
            pseudo_aus_list.append(str(pseudo_aus[au_idx]))
        pseudo_aus_str = ",".join(pseudo_aus_list)

        return pseudo_aus_str

    def gen_by_table_one(self, au_idx, aus_prob, pick_num): 
        if au_idx in aus_prob:
            if aus_prob[au_idx] >= 0.7:
                return int(pick_num < random.uniform(0.7, 1.0)) # sample probability in [0.7, 1]
            else:
                return int(pick_num < aus_prob[au_idx]) # sample by the given probability
        else:
            return int(pick_num < random.uniform(0., 0.2)) # sample probability in [0., 0.2]

    def meet_table_two(self, au_idx, aus_relation, exist_aus):
        if au_idx in aus_relation:
            cur_depend = aus_relation[au_idx]
            for x_items in cur_depend: 
                # single dependent
                if (len(x_items) == 1) and (x_items[0] in exist_aus) and (exist_aus[x_items[0]] == 1):
                    return True
                else:  # multiple dependent
                    for y_item in x_items:
                        if not ((y_item in exist_aus) and (exist_aus[y_item] == 1)):
                            return False
        return False

    def meet_table_three(self, au_idx, aus_depend, exist_aus):
        if au_idx in aus_depend:
            for item in aus_depend[au_idx]:
                if (item in exist_aus) and (exist_aus[item] == 1):
                    return True
        return False

    def write_to_file(self, saved_log):
        with open(self.saved_path, 'w') as f:
            file_head = ["Expression"]
            file_head.extend(list(map(lambda x: "AU%d" % x, self.aus_list)))
            f.write(",".join(file_head) + "\n")

            f.write("\n".join(saved_log))

def main():
    genPseudoAU = GenPseudoAU()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--aus', type=str, default='1,2,4,5,6,7,9,12,17,23,24,25', help='AUs vector index.')
    parser.add_argument('--n_sample', type=int, default=200, help='number of sample for per expression.')
    parser.add_argument('--saved_path', type=str, default="ck_pseudo_aus.csv", help='Save results to saved_path.')
    opt = parser.parse_args()

    genPseudoAU.initialize(opt)
    genPseudoAU.run()


if __name__ == "__main__":
    main()
