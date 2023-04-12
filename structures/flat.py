import csv

#не нужна пока, убрать, если функция пригодилась где-то
def csvWriter(filename, flat_list_dict):

    with open(filename, 'wt',encoding='utf-8') as fout:
        csvout = csv.DictWriter(fout,['cost', 'total_square', 'live_square', 'rooms_num', 'floor',
                                    'total_floors', 'metro', 'm_distance', 'adr', 'date_time', 'desc', 'house_type'])
        csvout.writeheader()
        csvout.writerows(flat_list_dict)

           # for item in flat_list_dict: csvout.writerow({k:v.encode('utf-8') for k, v in item.items()})




