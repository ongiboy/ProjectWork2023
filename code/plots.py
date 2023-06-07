import matplotlib.pyplot as plt

#### PRETRAINING ####
loss=[6.602673530578613, 4.967989921569824, 3.129988670349121, 2.185349702835083, 1.41599440574646, 0.5632876753807068, 0.44269782304763794, 0.008677244186401367, -0.7173727750778198, -0.9481281042098999, -1.1384022235870361, -1.5056321620941162, -1.6870415210723877, -1.8850895166397095, -2.020575761795044, -2.094316005706787, -2.2815871238708496, -2.2317681312561035, -2.1973683834075928, -2.2109289169311523, -2.412091016769409, -2.183976650238037, -2.542304515838623, -2.564032554626465, -2.397710084915161, -2.6258625984191895, -2.5076026916503906, -2.6112983226776123, -2.764944314956665, -2.8143157958984375, -2.854027271270752, -2.838191270828247, -2.9089198112487793, -2.954604387283325, -2.865999698638916, -2.85437273979187, -2.9252634048461914, -3.0818114280700684, -3.076387882232666, -3.1078147888183594]

loss_t= [5.781157493591309, 5.669548034667969, 5.57801628112793, 5.5096049308776855, 5.308920860290527, 5.148505210876465, 5.028321266174316, 4.920528411865234, 4.885912895202637, 4.816252708435059, 4.865638732910156, 4.806368350982666, 4.7706708908081055, 4.7673139572143555, 4.730696678161621, 4.713562965393066, 4.69368839263916, 4.656421661376953, 4.681227207183838, 4.6124467849731445, 4.593533992767334, 4.609128475189209, 4.566638946533203, 4.536378383636475, 4.55855655670166, 4.549903869628906, 4.524873733520508, 4.512871742248535, 4.491145610809326, 4.495062828063965, 4.494779586791992, 4.49142599105835, 4.470836639404297, 4.477010250091553, 4.444549083709717, 4.449582576751709, 4.441926002502441, 4.412160396575928, 4.421515464782715, 4.400692939758301]

loss_f= [5.989348888397217, 5.897663593292236, 5.901786804199219, 5.9556965827941895, 6.015070915222168, 6.070101737976074, 6.055943489074707, 6.161064624786377, 6.170066833496094, 6.117082118988037, 6.063257217407227, 6.033281326293945, 6.001456260681152, 5.969536304473877, 5.940835475921631, 5.914648056030273, 5.899472236633301, 5.884068965911865, 5.865335464477539, 5.846380233764648, 5.819937229156494, 5.82358980178833, 5.797306537628174, 5.769992828369141, 5.752484321594238, 5.751807689666748, 5.739742279052734, 5.721153259277344, 5.691427707672119, 5.66898250579834, 5.675074577331543, 5.657488822937012, 5.634474754333496, 5.619259834289551, 5.626202583312988, 5.622343063354492, 5.605215549468994, 5.592852592468262, 5.57688045501709, 5.551247596740723]

loss_c= [1.4348385334014893, -1.6312328577041626, -5.219825267791748, -7.094603061676025, -8.492002487182617, -10.092031478881836, -10.198867797851562, -11.064238548278809, -12.490724563598633, -12.829591751098633, -13.205700874328613, -13.850914001464844, -14.146209716796875, -14.507028579711914, -14.712681770324707, -14.81684398651123, -15.156335830688477, -15.004026412963867, -14.94129753112793, -14.880685806274414, -15.237653732299805, -14.80066967010498, -15.448554992675781, -15.434435844421387, -15.106460571289062, -15.553436279296875, -15.279820442199707, -15.456621170043945, -15.712461471557617, -15.79267692565918, -15.877908706665039, -15.825296401977539, -15.923151016235352, -16.00547981262207, -15.802752494812012, -15.780672073364258, -15.897666931152344, -16.168636322021484, -16.151172637939453, -16.16756820678711]

loss_TF= [8.225397109985352, 7.941378116607666, 6.990170955657959, 6.809386253356934, 6.596230983734131, 6.516288757324219, 6.117062568664551, 6.405384063720703, 5.891641139984131, 5.902074813842773, 5.9162774085998535, 5.720659255981445, 5.74895715713501, 5.699631214141846, 5.602047443389893, 5.676780700683594, 5.580422878265381, 5.54079532623291, 5.800373077392578, 5.602035999298096, 5.5134992599487305, 5.866214752197266, 5.636675834655762, 5.508779048919678, 5.782910346984863, 5.633762359619141, 5.560847759246826, 5.68596887588501, 5.513280391693115, 5.481168270111084, 5.5482587814331055, 5.546323776245117, 5.407857418060303, 5.439481258392334, 5.523785591125488, 5.519375801086426, 5.49793004989624, 5.419931888580322, 5.4133477210998535, 5.375662803649902]

### FINE TUNING ###
finetune_acc = [0.5, 0.5, 0.5166666507720947, 0.6000000238418579, 0.6333333253860474, 0.699999988079071, 0.6666666865348816, 0.7166666388511658, 0.7166666388511658, 0.699999988079071, 0.7333333492279053, 0.6833333373069763, 0.7333333492279053, 0.75, 0.7833333611488342, 0.7333333492279053, 0.75, 0.75, 0.800000011920929, 0.75, 0.800000011920929, 0.7666666507720947, 0.75, 0.7833333611488342, 0.800000011920929, 0.7333333492279053, 0.7666666507720947, 0.8333333134651184, 0.7166666388511658, 0.800000011920929, 0.7833333611488342, 0.7833333611488342, 0.75, 0.7666666507720947, 0.75, 0.75, 0.7166666388511658, 0.7666666507720947, 0.7833333611488342, 0.800000011920929]

finetune_loss = [7.868227005004883, 7.654852390289307, 7.013343811035156, 7.965806007385254, 7.33580207824707, 7.751387596130371, 7.051374435424805, 7.38874626159668, 7.599369049072266, 7.201810836791992, 7.382058143615723, 7.03814172744751, 7.4660797119140625, 7.468334674835205, 6.566958427429199, 7.0824079513549805, 7.796591281890869, 7.184172630310059, 7.856825351715088, 7.348228454589844, 6.811399459838867, 7.70804500579834, 7.013994216918945, 7.41478157043457, 7.3929877281188965, 7.023088455200195, 7.048946380615234, 7.511425971984863, 6.962553977966309, 7.397516250610352, 7.058960914611816, 7.533249855041504, 6.996395111083984, 7.444860935211182, 7.677136421203613, 7.589941024780273, 7.817572593688965, 7.19706392288208, 7.225527763366699, 7.3141913414001465]

test_acc = [0.7990000247955322, 0.7989999651908875, 0.7990000247955322, 0.7990000247955322, 0.7989999651908875, 0.7990000247955322, 0.7990000247955322, 0.7989999651908875, 0.7990000247955322, 0.799000084400177, 0.7990000247955322, 0.7990000247955322, 0.7990000247955322, 0.7990000247955322, 0.7990000247955322, 0.7990000247955322, 0.7989999651908875, 0.7989999651908875, 0.7990000247955322, 0.7989999651908875, 0.7990000247955322, 0.7990000247955322, 0.7990000247955322, 0.7989999651908875, 0.7989999651908875, 0.7989999651908875, 0.7990000247955322, 0.7990000247955322, 0.7990000247955322, 0.7989999651908875, 0.7989999651908875, 0.7990000247955322, 0.7989999651908875, 0.7990000247955322, 0.7990000247955322, 0.7989999651908875, 0.7990000247955322, 0.7990000247955322, 0.7990000247955322, 0.7989999651908875]

test_loss = [0.6254710555076599, 0.6254710555076599, 0.6206724047660828, 0.6142730116844177, 0.6064486503601074, 0.5982453227043152, 0.59824538230896, 0.5900506377220154, 0.5826254487037659, 0.5826255083084106, 0.5760118365287781, 0.5760118365287781, 0.5703679919242859, 0.5671920776367188, 0.5666619539260864, 0.5666619539260864, 0.5666619539260864, 0.5666620135307312, 0.5663976669311523, 0.5663976073265076, 0.5670113563537598, 0.567011296749115, 0.5670113563537598, 0.5670113563537598, 0.5670113563537598, 0.5670113563537598, 0.5670113563537598, 0.5682910084724426, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910680770874, 0.5682910084724426, 0.5682910680770874]

################## Training is Done! #######################


fig, axs = plt.subplots(2, 3, figsize=(8, 6))

# Plot data on each subplot
axs[0, 0].plot(loss)
axs[0, 1].plot(loss_t)
axs[1, 0].plot(loss_f)
axs[1, 1].plot(loss_c)
axs[1, 2].plot(loss_TF)

# Add titles to subplots
axs[0, 0].set_title('Average loss')
axs[0, 1].set_title('Average loss time')
axs[1, 0].set_title('Average loss frequency')
axs[1, 1].set_title('Average loss c')
axs[1, 2].set_title('Average loss TF')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


##### Plot finetune #####
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

# Plot data on each subplot
axs[0, 0].plot(finetune_acc)
axs[0, 1].plot(finetune_loss)
axs[1, 0].plot(test_acc)
axs[1, 1].plot(test_loss)

# Add titles to subplots
axs[0, 0].set_title('Finetune accuracy')
axs[0, 1].set_title('Finetune loss')
axs[1, 0].set_title('Test accuracy')
axs[1, 1].set_title('Test loss')


# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()