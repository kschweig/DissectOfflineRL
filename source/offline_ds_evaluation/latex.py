

def create_latex_table(path, arguments):

    environment = {"MiniGrid-LavaGapS6-v0": "lava", "MiniGrid-SimpleCrossingS9N1-v0": "simple",
                   "CartPole-v1": "cartpole", "Acrobot-v1":"acrobot", "MountainCar-v0": "mountaincar"}
    buffer = {"er": "Experience Replay", "fully": "Final Policy", "random": "Random Policy",
              "mixed": "Mixed Policy", "noisy": "Noisy Final Policy"}
    results = ["Reward", "Reward (Normalized)", "Entropy (Normalized)", "Episode Length", "Sparsity",
               "Unique States / Episode", "Uniqueness", "Unique States"]

    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\begin{tabular}{l|" + "c"*len(arguments) + "}\n")

        f.write("Metric  \\hspace{8pt} \\symbol{92} \\hspace{8pt} Buffer Type")
        for i in range(len(arguments)):
            f.write(" & " + buffer[arguments[i][1]])
        f.write(" \\\\ \\hline \n")

        for j in range(2, len(arguments[0])):
            f.write(results[j-2] + " & ")
            for i in range(len(arguments)):
                if isinstance(arguments[i][j], tuple):
                    f.write(f"${round(arguments[i][j][0], 2):.2f} \\pm {round(arguments[i][j][1], 2):.2f}$")
                else:
                    f.write(f"${round(arguments[i][j], 5)}$")
                if i == len(arguments) - 1:
                    f.write("\\\\ \n")
                else:
                    f.write(" & ")

        f.write("\\end{tabular}\n\\caption{Dataset evaluation metrics for all buffer types of environment '"
                +arguments[0][0]+"'.}\n")
        f.write("\\label{tab:ds_eval_"+environment[arguments[0][0]]+"}\n\\end{table}")