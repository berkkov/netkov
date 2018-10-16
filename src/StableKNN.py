import torch
from Sampler import prepare_K_test



def prepare_distance_dict():
    crop_d = torch.load("D:\\adam4375_crop_fv.pth", map_location='cpu')
    catalog_d = torch.load("D:\\adam4375_cat_fv.pth", map_location='cpu')
    print(len(crop_d))
    print(len(catalog_d))
    truth, test_crop_paths = prepare_K_test('dresses')

    distance_by_crop = {}
    count = 0
    test_ids = torch.load("D:\\netkov\\Netkov\\test_ids.pkl")

    for m in crop_d:
        i = m.split('\\')[-1]
        if i not in truth:      # test_ids
            count += 1
            continue

        distance_by_crop[i] = []
        anchor_fv = crop_d[m]
        count += 1
        print((count)/10350 * 100)
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
            if check in netkov_out[a][:K]:
                corrects += 1
                break

    print("Top-" + str(K) + "-Accuracy: " + str(corrects / len(truth)))
    print(corrects)
    return corrects / len(truth)

k = 3000
top_K_test(k, 'dresses', output_path='D:\\adam4375_knn_id.pkl')
#top_K_test(k, 'dresses', output_path='D:\\adam3100_knn_id.pkl')
#top_K_test(k, 'dresses', output_path='D:\\adam2950_knn_id.pkl')
#top_K_test(k, 'dresses', output_path='D:\\adam2325_knn_id.pkl')
#top_K_test(k, 'dresses', output_path='D:\\adam1375_knn_id.pkl')

#dist_dict = prepare_distance_dict()
#rank_kNN(3000, dist_dict, 'D:\\adam4375_knn.pkl')
#print(dist_dict)


structure = False
whole_dict = {}
if structure:
    '''
    for i in range(0, 2):
        filename = "D:\\knn_clean22_" + str(i) + ".pkl"
        dict2 = torch.load(filename)
        dict = {}
        
        # For DRIVE ONLY
        for key in dict2:
            dict[key.split('/')[-1].split('.')[0]] = [(x[0].split('/')[-1].split('.')[0], x[1]) for x in dict2[key]]
        for key in dict:
            whole_dict[key] = dict[key]

    torch.save(whole_dict, "D:\\knn_clean22.pkl")
    print(whole_dict)
    print(len(whole_dict))
    '''
    dict = torch.load('D:\\adam4375_knn.pkl')
    only_id = {}
    for i in dict:
        only_id[i]= []
        for val in dict[i]:
            only_id[i].append(val[0])
    torch.save(only_id, 'D:\\adam4375_knn_id.pkl')
    print(len(only_id))

#print(torch.load('D:\\netkov\\Netkov\\test_ids.pkl'))
#print(torch.load("D:\\cropset_knn_8999.pkl")['7683'])
#print(torch.load("D:\\cropset_knn_8999.pkl")['9251'])

'''
if __name__ == '__main__':

    #prepare_distance_dict()
    #top_K_test(25, 'dresses')
    crop_d = torch.load("D:\\adam4375_crop_fv.pth", map_location='cpu')
    catalog_d = torch.load("D:\\adam4375_cat_fv.pth", map_location='cpu')

    distance_by_crop = {}
    count = 0
    for i in crop_d:
        if i not in ['8878']:
            continue
        count += 1
        if count == 20:
            break
        distance_by_crop[i] = []
        anchor_fv = crop_d[i]
        print(count / len(crop_d) * 100)
        for k in catalog_d:
            cat_fv = catalog_d[k]
            dist = torch.dist(anchor_fv, cat_fv)
            distance_by_crop[i].append((k, dist.item()))


    for i in distance_by_crop.keys():
        distance_by_crop[i] = sorted(distance_by_crop[i], key=lambda cat_img: cat_img[1])

    print(distance_by_crop['8878'])

    

    #222222222222222222222222222222222222222222222222222222222222
    # prepare_distance_dict()
    # top_K_test(25, 'dresses')
    crop_d = torch.load("D:\\3_4000_test\\crop_fv_3_4000.pth", map_location='cpu')
    catalog_d = torch.load("D:\\3_4000_test\\catalog_fv_3_4000.pth", map_location='cpu')

    distance_by_crop = {}
    count = 0
    for i in crop_d:
        if i not in ['9836']:
            continue
        count += 1
        if count == 20:
            break
        distance_by_crop[i] = []
        anchor_fv = crop_d[i]
        print(count / len(crop_d) * 100)
        for k in catalog_d:
            cat_fv = catalog_d[k]
            dist = torch.dist(anchor_fv, cat_fv)
            distance_by_crop[i].append((k, dist.item()))

    for i in distance_by_crop.keys():
        distance_by_crop[i] = sorted(distance_by_crop[i], key=lambda cat_img: cat_img[1])
    print('\n')
    #print(distance_by_crop['9836'])








    # 3333333333333333333333333333333333333333333333333333333333333333
    # prepare_distance_dict()
    # top_K_test(25, 'dresses')
    crop_d = torch.load("D:\\crop_fv_41_7000.pth", map_location='cpu')
    catalog_d = torch.load("D:\\catalog_fv_41_7000.pth", map_location='cpu')

    distance_by_crop = {}
    count = 0
    for i in crop_d:
        if i not in ['9836']:
            continue
        count += 1
        if count == 20:
            break
        distance_by_crop[i] = []
        anchor_fv = crop_d[i]
        print(count / len(crop_d) * 100)
        for k in catalog_d:
            cat_fv = catalog_d[k]
            dist = torch.dist(anchor_fv, cat_fv)
            distance_by_crop[i].append((k, dist.item()))

    for i in distance_by_crop.keys():
        distance_by_crop[i] = sorted(distance_by_crop[i], key=lambda cat_img: cat_img[1])
    print('\n')
    print(distance_by_crop['9836'])


'''
'''
dict = torch.load('D:\\cropset_knn_2_6999.pkl')
print(dict['16319'])

negs for 16319
'59368', 0.5821760296821594
('62327', 0.6646803617477417)
('45968', 0.7524970769882202)
('32365', 0.5748361349105835
('59368', 0.5821760296821594)

pos for 16319
('37491', 0.46807825565338135)
('37492', 0.4821515381336212)
('37493', 0.5279341340065002)
('37494', 0.38954612612724304)
'''
