import csv

dictionaries = [{'age': '30', 'roi': '[[1,2][3,4]]', 'last_name': 'Doe'}, {'age': '30', 'roi': '[[5,6]]', 'last_name': 'Doe'}]
with open('my.csv', 'w+') as csv_file:
    headers = [k for k in dictionaries[0]]
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()
    for dictionary in dictionaries:
        writer.writerow(dictionary)

# csv_file=open('my.csv', 'r+')
# reader = csv.DictReader(csv_file)
# for row in reader:
#     print(row['roi'])
        
with open('my.csv', 'r+') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        print(row['roi'])


