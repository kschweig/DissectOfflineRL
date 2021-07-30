

def create_latex_table(path, arguments):

    environment = {"MiniGrid-LavaGapS7-v0": "lava", "MiniGrid-Dynamic-Obstacles-8x8-v0": "obstacles",
                   "CartPole-v1": "cartpole", "Acrobot-v1":"acrobot", "MountainCar-v0": "mountaincar",
                   "Breakout-MinAtar-v0": "breakout", "Space_invaders-MinAtar-v0": "spaceinvaders"}
    buffer = { "random": "Random Policy", "mixed": "Mixed Policy", "er": "Exp. Replay",
              "noisy": "Noisy Policy", "fully": "Final Policy"}
    results = ["Return", "Unique State-Action Pairs", "Entropy"]

    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\begin{tabular}{l" + "c"*len(results) + "}\n \hline \n")

        f.write("Buffer Type")
        for result in results:
            f.write(" & " + result)
        f.write(" \\\\ \\hline \n")

        for i, buf in enumerate(buffer.values()):
            f.write(buf + " & ")
            for j in range(2, len(arguments)):

                if isinstance(arguments[i][j], tuple):
                    f.write(f"${round(arguments[i][j][0], 2):.2f} \\pm {round(arguments[i][j][1], 2):.2f}$")
                else:
                    f.write(f"${round(arguments[i][j], 5)}$")
                if j == len(arguments) - 1:
                    f.write("\\\\ \n")
                else:
                    f.write(" & ")

        f.write("\n \hline \n")
        f.write("\\end{tabular}\n\\caption{Dataset evaluation metrics for all buffer types of environment '"
                +arguments[0][0].replace("_", "\_") +"'.}\n")
        f.write("\\label{tab:ds_eval_"+environment[arguments[0][0]]+"}\n\\end{table}")