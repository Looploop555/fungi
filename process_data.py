

import os
import csv
import pickle


def load_csv(path):

    reader = csv.reader(open(path, 'r'))
    csv_title = next(reader)
    csv_data = [row for row in reader]
    return csv_title, csv_data


def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def csv2dict(head, data, index_flag="category_id"):
    zhenjun = {}
    species = {}
    for i, line in enumerate(data):
        index = head.index(index_flag)        
        zhenjun[line[index]] = zhenjun.get(line[index], [])  
        info = {}
        for idx, clas in enumerate(head):
            info[clas] = line[idx] if clas != "landcover" else get_land_cover_categories(line[idx], line)
            
            if clas == "species":
                if species.get(line[idx], None) is None:
                    species[line[idx]] = line[index]
                elif species[line[idx]] != line[index]:
                    print("species: %s, old category_id: %s, category_id: %s" % (line[idx], species[line[idx]], line[index]))


        zhenjun[line[index]].append(info)
    return zhenjun


def dict_mapping_dir(categories, image_dir):
    new_categories = {}
    for k, v in categories.items():
        new_categories[k] = {
            "old_info": v,
            "image_urls": [os.path.join(image_dir, f["filename"]) for f in v]
        }
    return new_categories


if __name__ == "__main__":
    image_dir = "/data/fungi/images/FungiTastic-FewShot/train/fullsize"
    csv_path = "/data/fungi/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv"
    head, data = load_csv(csv_path)
    zhenjun_dict = csv2dict(head, data, index_flag="category_id")   #
    zhenjun_dict = dict_mapping_dir(zhenjun_dict, image_dir)
    # zhenjun_dict.pop("-1")
    save_pickle(
        zhenjun_dict,
        "FungiTastic-FewShot-Train.pkl"
    )
