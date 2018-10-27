import torch
import os
import json


def prepare_K_test(category):

    truth = {}
    base_dir = "D:\\netkov\\data"
    meta_dir = os.path.join(base_dir, "meta", "json")
    retrieval_path = os.path.join(meta_dir, "retrieval_" + category + ".json")
    base_img_dir = os.path.join(base_dir, "structured")
    query_dir = os.path.join(base_img_dir, "wtbi_" + category + "_query_crop")
    catalog_dir = os.path.join(base_img_dir, category + "_catalog")


    filename = "test_pairs_" + category + ".json"
    photo_to_product_map = {}
    with open(os.path.join(meta_dir, filename)) as jsonFile:
        pairs = json.load(jsonFile)

    with open(retrieval_path) as jsonFile:
        data = json.load(jsonFile)

    for info in data:
        photo_to_product_map[info['photo']] = info['product']

    product_to_photo_map = {}

    for photo in photo_to_product_map:
        product = photo_to_product_map[photo]
        if product not in product_to_photo_map:
            product_to_photo_map[product] = set()

        product_to_photo_map[product].add(photo)

    for pair in pairs:
        photo = pair["photo"]
        if not os.path.exists(os.path.join(query_dir, str(photo) + ".jpg")):
            continue
        product = pair["product"]

        p_s = []
        for positive in product_to_photo_map[product]:
            if not os.path.exists(os.path.join(catalog_dir, str(positive) + ".jpg")):
                continue
            p_s.append(positive)

        truth[str(photo)] = p_s
        if len(truth[str(photo)]) == 0:
            del truth[str(photo)]

    test_crop_paths = [os.path.join(query_dir,str(x) + '.jpg') for x in truth.keys()]

    return truth, test_crop_paths


def prepare_distance_dict(run_id, vertical):
    crop_path = "D:\\Indirilenler\\" + run_id + "_crop_fv.pth"
    catalog_path = "D:\\Indirilenler\\" + run_id + "_cat_fv.pth"


    crop_d = torch.load(crop_path, map_location='cpu')
    catalog_d = torch.load(catalog_path, map_location='cpu')
    print("Number of crop images:", len(crop_d))
    print("Number of catalog images:", len(catalog_d))
    truth, test_crop_paths = prepare_K_test(vertical)

    distance_by_crop = {}
    count = 0
    test_ids = torch.load("D:\\netkov\\Netkov\\test_ids.pkl")

    for m in crop_d:
        i = m.split('\\')[-1]
        if i not in test_ids or i not in truth:      # test_ids
            count += 1
            continue

        distance_by_crop[i] = []
        anchor_fv = crop_d[m]
        count += 1
        print((count)/len(crop_d) * 100)
        for k in catalog_d:
            cat_fv = catalog_d[k]
            dist = torch.dist(anchor_fv, cat_fv)
            distance_by_crop[i].append((k, dist.item()))

    return distance_by_crop

def rank_kNN(K, distance_dict, out_path='cropset_knn_1_399.pkl'):
    for i in distance_dict.keys():
        distance_dict[i] = sorted(distance_dict[i], key=lambda cat_img: cat_img[1])[:K]

    torch.save(distance_dict, out_path)
    return distance_dict


def top_K_test(K, category, output_path):


    truth, test_crop_paths = prepare_K_test(category)
    print(len(truth))
    with open(output_path, "rb") as knnFile:
        model_results = torch.load(knnFile)

    print('load results')
    netkov_out = {}
    for k in model_results:
        netkov_out[int(k)] = []
        for no in model_results[k]:
            netkov_out[int(k)].append(int(no))

    corrects = 0
    count = 0
    print('Truth check')
    for id in truth:
        for check in truth[id]:
            count +=1
            a = int(id)
            if a in netkov_out:
                if check in netkov_out[a][:K]:
                    corrects += 1
                    break

    print("Top-" + str(K) + "-Accuracy: " + str(corrects / len(truth)))
    print(corrects)
    return corrects / len(truth)


def run_knn(run_id, k=10, vertical='dresses', max_k=10000):
    structured_path = run_id + "_knn_id.pkl"
    if not os.path.exists(structured_path):
        dist_dict = prepare_distance_dict(run_id, vertical)
        knn_out_path = run_id + "_knn.pkl"
        rank_kNN(max_k, dist_dict, knn_out_path)

        dict = torch.load(knn_out_path)
        only_id = {}
        for i in dict:
            only_id[i] = []
            for val in dict[i]:
                only_id[i].append(val[0])
        torch.save(only_id, structured_path)
        print(len(only_id))
    top_K_test(k, vertical, structured_path)
    return


if __name__ == '__main__':
    run_id = "nesterov825"
    run_knn(run_id)





