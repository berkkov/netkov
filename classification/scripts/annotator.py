import os
import glob
import torch

# PART 1 -- AMAZON DATA
# Training
base_path = 'C:\\Users\\user\\Desktop\\netkov_data\\all\\'
os.chdir(base_path)
distinct_dress = glob.glob('*')

tr_proxy_id = 0
proxy_id_to_img_path = {}
for distinct_dress_id in distinct_dress:
    catalogs = [cat for cat in glob.glob(f'{distinct_dress_id}\\*') if '.jpg' not in cat]
    if len(catalogs) == 0:
        imgs = glob.glob(f'{distinct_dress_id}\\*.jpg')
        proxy_id_to_img_path[tr_proxy_id] = imgs
        tr_proxy_id += 1
    else:
        for _cat in catalogs:
            imgs = glob.glob(f'{_cat}\\*.jpg')
            proxy_id_to_img_path[tr_proxy_id] = imgs
            tr_proxy_id += 1


less_than_4 = []
sum = 0
for dress in proxy_id_to_img_path:
    if len(proxy_id_to_img_path[dress]) < 4:
        less_than_4.append(dress)
        sum += len(proxy_id_to_img_path[dress])

sum = 0
for k in less_than_4:
    if 'cat' in proxy_id_to_img_path[k][0]:
        sum += 1
len(less_than_4) - sum
len(less_than_4)

test_proxies = less_than_4
train_proxies = [proxy_id for proxy_id in proxy_id_to_img_path if proxy_id not in test_proxies]

train_proxy_to_img_path = {}
test_proxy_to_img_path = {}
for new_idx, old_idx in enumerate(train_proxies):
    train_proxy_to_img_path[new_idx] = proxy_id_to_img_path[old_idx]
for new_idx, old_idx in enumerate(test_proxies):
    test_proxy_to_img_path[new_idx] = proxy_id_to_img_path[old_idx]

img_to_label_amazon_train = {}
img_to_label_amazon_test = {}
for label in train_proxy_to_img_path:
    for img in train_proxy_to_img_path[label]:
        img_to_label_amazon_train[img] = label
for label in test_proxy_to_img_path:
    for img in test_proxy_to_img_path[label]:
        img_to_label_amazon_test[img] = label

torch.save(img_to_label_amazon_train,
           'C:\\Users\\user\\Desktop\\anno\\amazon_train_img_to_label.pkl')
torch.save(img_to_label_amazon_test,
           'C:\\Users\\user\\Desktop\\anno\\amazon_test_img_to_label.pkl')

# END PART 1
# PART 2
# Add Query - Catalog dataset
base_path = 'C:\\Users\\user\\Downloads\\netkov_data_2\\'
os.chdir(base_path)
distinct_dress = glob.glob('*')

tr_proxy_id = 0
proxy_id_to_img_path = {}
for distinct_dress_id in distinct_dress:
    imgs = glob.glob(f'{distinct_dress_id}\\*.jpg') + glob.glob(f'{distinct_dress_id}\\*\\*.jpg')
    proxy_id_to_img_path[tr_proxy_id] = imgs
    tr_proxy_id += 1


less_than_4 = []
sum = 0
for dress in proxy_id_to_img_path:
    if len(proxy_id_to_img_path[dress]) < 3:
        less_than_4.append(dress)
        sum += len(proxy_id_to_img_path[dress])

for k in less_than_4:
    if 'cat' in proxy_id_to_img_path[k][0]:
        sum += 1

print(less_than_4)
test_proxies = less_than_4
train_proxies = [proxy_id for proxy_id in proxy_id_to_img_path if proxy_id not in test_proxies]
proxy_id_to_img_path

# PART 2 B || Creating a conventional test here: query-to-catalog || Use everything as test
query_to_catalog = {}
for proxy in proxy_id_to_img_path:
    catalog = [x for x in proxy_id_to_img_path[proxy] if 'cat' in x]
    queries = [x for x in proxy_id_to_img_path[proxy] if 'cat' not in x]
    if len(catalog) == 0:
        continue
    for query_img in queries:
        query_to_catalog[query_img] = catalog
query_to_catalog # TODO: SAVE (Ann. 2.7) torch.save(query_to_catalog, 'qap_query_to_catalog.pkl')

torch.save(query_to_catalog, 'C:\\Users\\user\\Desktop\\anno\\qap_query_to_catalog.pkl')
list(query_to_catalog.values())
# For query images img_id : img_path
qap_query_id_to_path = dict(zip([x.split('\\')[-1].split('.')[0] for x in query_to_catalog.keys()],
                                query_to_catalog.keys())
                            )  # TODO: SAVE (Ann 2.1)


qap_query_path_to_id = {v: k for k, v in qap_query_id_to_path.items()}  # TODO: SAVE (Ann 2.2)

# For catalog images img_id: img_path
qap_catalog_img_list = []
for k in query_to_catalog.values():
    qap_catalog_img_list += k
qap_catalog_img_list = list(set(qap_catalog_img_list))


qap_catalog_id_to_path = dict(zip([x.split('\\')[-1].split('.')[0] for x in qap_catalog_img_list],
                                  qap_catalog_img_list)
                              )  # TODO: SAVE (Ann 2.3)


qap_catalog_path_to_id = {v: k for k, v in qap_catalog_id_to_path.items()}  # TODO: SAVE (Ann 2.4)
torch.save(qap_catalog_path_to_id, 'C:\\Users\\user\\Desktop\\anno\\qap_catalog_path_to_id.pkl')

# PART 3
# Add catalog crowders for test
os.chdir('C:\\Users\\user\\Downloads\\dresses_catalog\\')

crowder_img_path_to_name = {crowder_path: crowder_path.split('.')[0].split('\\')[-1]
                            for crowder_path in glob.glob('*\\*.jpg')}

test = torch.load('C:\\Users\\user\\Desktop\\anno\\qap_catalog_path_to_id.pkl')

ban_list = []
for k in list(crowder_img_path_to_name.values()):
    if k in list(test.values()):
        ban_list.append(k)


crowder_img_path_to_name = {k: v for k, v in crowder_img_path_to_name.items() if v not in ban_list}
torch.save(crowder_img_path_to_name, 'C:\\Users\\user\\Desktop\\anno\\crowder_img_path_to_name.pkl')

