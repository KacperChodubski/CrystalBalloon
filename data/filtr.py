import csv

cat = set()
uniq = list()
wantedRecords = []
firstRow = ""
# with open("/Users/mojskarb/Desktop/loon-flights-2011-2014.csv", newline='') as file:
#     reader = csv.reader(file, dialect='excel',  delimiter=',')
#     for line in reader:
#         cat.add(line[0])

#     for name in cat:
#         cos = name.split('-')[0]
#         if cos not in uniq:
#             uniq.append(cos)
    
with open("/Users/mojskarb/Desktop/loon-flights-2011-2014.csv", newline='') as file3:
    reader = csv.reader(file3, dialect='excel',  delimiter=',')
    firstRow = next(reader)
    for line in reader:
        cos = line[0].split('-')[0]
        if cos == "Icarus" or cos == "PT":
            wantedRecords.append(line)

print(len(wantedRecords))

with open("/Users/mojskarb/Desktop/moje.csv", 'w', newline='') as file2:
    writer = csv.writer(file2)
    writer.writerow(firstRow)
    for row in wantedRecords:
        print(row)
        writer.writerow(row)




        
