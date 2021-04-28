from tensorflow.python.summary.summary_iterator import summary_iterator

###########################
# load tb logs from files #
###########################

for e in summary_iterator("../../runs/ex5"):
    for v in e.summary.value:
        if v.tag == 'reward':
            print(v.simple_value)