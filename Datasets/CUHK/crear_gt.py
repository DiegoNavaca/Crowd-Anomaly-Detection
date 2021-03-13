import pandas

datos = pandas.read_excel("info.xls")

f = open("ground_truth.txt","w")

for i in range(len(datos["video_name"])):
    f.write(str(datos["video_name"][i])+","+str(datos["video_gt"][i])+"\n")

f.close()


