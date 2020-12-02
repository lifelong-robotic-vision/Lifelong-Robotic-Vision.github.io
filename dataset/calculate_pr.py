from tools.main import *
import cv2
import matplotlib.pyplot as plt
from config.config import opt


def evaluate(mode='detect'):
    # Initialization: loading tree
    tree = Tree(opt.K, opt.L, None)
    tree.load_tree(opt.voc_file)
    matcher = Matcher(tree)

    Ps = []
    Rs = []

    P_R = []

    if mode == 'detect':
        """
            calculating similarity.
        """
        # get similarity and graph verification score
        scores, veri, _ = detect_loops(tree, matcher,
                                          opt.testing_set,
                                          threshold=1.0)

        # save similarity
        with open(opt.score_file, 'w') as f:
            for i in range(tree.N):
                for j in range(tree.N):
                    f.write(str(scores[i, j]))
                    f.write(' ')
                f.write('\n')
        # save verification score
        with open(opt.gv_file, 'w') as f:
            for i in range(tree.N):
                for j in range(tree.N):
                    f.write(str(veri[i, j]))
                    f.write(' ')
                f.write('\n')
    elif mode == 'show':
        """
            calculate precision and recall.
        """
        # load similarity
        scores = np.zeros((tree.N, tree.N), dtype=float)
        with open(opt.score_file, 'r') as f:
            lines = f.readlines()
            # print(len(lines))
            for (i, line) in enumerate(lines):
                infos = line.split(' ')
                infos.remove('\n')
                # print(len(scores))
                for (j, score) in enumerate(infos):
                    scores[i, j] = eval(score)
        # load verification score
        veri = np.zeros((tree.N, tree.N), dtype=float)
        with open(opt.gv_file, 'r') as f:
            lines = f.readlines()
            # print(len(lines))
            for (i, line) in enumerate(lines):
                infos = line.split(' ')
                infos.remove('\n')
                # print(len(scores))
                for (j, score) in enumerate(infos):
                    veri[i, j] = eval(score)

        # thresholf = 0:0.02:0.999
        for thre in range(0, 999, 20):

            threshold = float(thre) / 1000.0
            print('==> Threshold: %f' % threshold)

            loops = []
            for i in range(tree.N):
                # # divide dataset into two parts: loop, query
                # if opt.dataset == 'NewCollege':
                #     if i * 2 < 188 or 370 < i * 2 < 552 or 632 < i * 2 < 1558 or 1730 < i * 2 < 1918:
                #         continue
                # elif opt.dataset == 'CityCentre':
                #     if i * 2 < 1352 or i * 2 >= 2428:
                #         continue
                # elif opt.dataset == 'KITTI06':
                #     if i < 835:
                #         continue

                res = {}
                for j in range(tree.N):
                    """
                        Only those image pairs whose similarity higher than threshold 
                        and verification score is higher than zero, will be regarded as 
                        loop closure candidates.
                    """
                    sim = scores[i, j]
                    gv = veri[i, j]
                    if sim >= threshold:
                        if gv > 0:
                            res[j] = 0.9 * sim + 0.1 * gv

                if res:
                    res = sorted(res.items(), key=lambda x: x[1], reverse=True)
                    # print(res)
                    res = np.array(res)
                    indexes = res[:, 0]
                    # Top 1
                    loops.append([i, int(indexes[0])])
            # print(loops)
            save_loops(loops, opt.result_file)
            loops = load_loops(opt.result_file)
            ground_truth_number, candidates_number, matched_number = get_pr(loops, opt   .ground_truth_file_path)

            if candidates_number == 0:
                break

            # calculate P,R
            P = matched_number / candidates_number
            R = matched_number / ground_truth_number

            print('==> P:%f, R:%f' % (P, R))
            Ps.append(P)
            Rs.append(R)
            P_R.append([P, R])

        # draw curve
        Ps = np.asarray(Ps)
        Rs = np.asarray(Rs)

        # from scipy.interpolate import spline
        plt.figure(1)
        plt.title('P-R curve')
        plt.plot(Rs, Ps, marker='o', linewidth=3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precise')
        plt.savefig(opt.fig_name)
        plt.show()
