######install Detectron2 using https://detectron2.readthedocs.io/en/latest/tutorials/install.html
import glob
import detectron2
from detectron2.utils.logger import setup_logger
from torch import nn
from detectron2.structures.instances import Instances
from utils import bb_intersection_over_union, showarray, doit, get_objects
import networkx as nx

setup_logger()
import os
from scipy.special import softmax
import cv2
import torch
import random
import pandas as pd
from sklearn.manifold import TSNE
from detectron2.structures.boxes import Boxes
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import warnings
from matplotlib import pyplot as plt
import io
import numpy as np
from IPython.display import clear_output, Image, display
import PIL.Image
from node2vec import Node2Vec
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
import warnings
import time
import scipy.sparse as sp
import torch.nn as nn
from DGI.models import DGI, LogReg
from DGI.utils import process
import plotly.express as px

warnings.filterwarnings("ignore", category=FutureWarning)
# Load VG Classes
data_path = 'data/genome/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes
image_name_list = []
for file in glob.glob("data/images/*.jpg"):
    image_name_list.append(file)
image_name_list.sort()
############################### total number of objects in the video
total_objects_num = 0
node_to_frame_mapping = []
total_time = 0
for img_indx in range(len(image_name_list)):
    print("image Number : ", str(img_indx))
    x_im = cv2.imread(image_name_list[img_indx])  # read input frame
    pred_boxes, scores, pred_class, pred, v, predictor = get_objects(x_im)
    v = v.draw_instance_predictions(pred)

    ################ Draw each object in an image
    # plt.imshow(v.get_image()[:, :, ::-1])
    # plt.show()

    for indx, obj_indx in enumerate(pred_boxes):
        a = int(obj_indx[0])
        b = int(obj_indx[1])
        c = int(obj_indx[2])
        d = int(obj_indx[3])

        cv2.imwrite("objects/" + str(indx + total_objects_num) + ".jpg", x_im[b:d, a:c, :])
        node_to_frame_mapping.append(img_indx)
    total_objects_num = total_objects_num + len(pred_boxes)
print("***************************Total object num: " + str(total_objects_num) + " total time " + str(total_time))

main_graph = np.zeros((total_objects_num, total_objects_num))  # Converted video into Graph
node_features = np.zeros((total_objects_num, 2048))  ### add features to each node
sum_of_previous_objects = 0
for img_indx in range(len(image_name_list) - 1):
    x_im = cv2.imread(image_name_list[img_indx])  # read input frame
    pred_boxes, scores, pred_class, pred, v, predictor = get_objects(x_im)
    a, tmp_feature_maps = doit(x_im, pred_boxes, predictor)
    for count in range(np.shape(tmp_feature_maps)[0]):
        node_features[sum_of_previous_objects + count, :] = tmp_feature_maps[count].cpu().view(-1)

    ########################## G_Space Matrix Genertaion
    print("number of objects in image : ", len(pred_boxes))
    dim_g_space = len(pred_boxes)
    G_Space = np.zeros((dim_g_space, dim_g_space))  # Spatial Feature Extraction
    for row in range(dim_g_space):
        for col in range(dim_g_space):
            G_Space[row, col] = bb_intersection_over_union(pred_boxes[row], pred_boxes[col])
    ############################
    G_Space_softmax = softmax(G_Space)
    v = v.draw_instance_predictions(pred)
    # Show the model architecture
    print(predictor.model)
    ########################################## G_time Matrix / Temporal Feature Extraction
    a, x_im_feature_maps = doit(x_im, pred_boxes, predictor)
    cos = nn.CosineSimilarity(dim=0)  # find similarity between objects between 2 Frames
    x_im_next = cv2.imread(image_name_list[img_indx + 1])
    pred_boxes_next, scores_next, pred_class_next, pred_next, v_next, predictor_next = get_objects(x_im_next)
    b, x_im_feature_maps_next = doit(x_im_next, pred_boxes_next, predictor_next)

    x_im_featurs_num = np.shape(x_im_feature_maps)[0]
    x_im_next_featurs_num = np.shape(x_im_feature_maps_next)[0]

    G_time_matrix = np.zeros((x_im_featurs_num, x_im_next_featurs_num))
    for i in range(x_im_featurs_num):
        for j in range(x_im_next_featurs_num):
            G_time_matrix[i][j] = cos(x_im_feature_maps[i].view(-1), x_im_feature_maps_next[j].view(-1)).item()
    G_time_matrix_softmax = softmax(G_time_matrix)
    ######### Spatio-Temporal Matrix Creation
    main_graph[sum_of_previous_objects:(sum_of_previous_objects + x_im_featurs_num),
    sum_of_previous_objects:(sum_of_previous_objects + x_im_featurs_num)] = G_Space_softmax[:, :]
    main_graph[sum_of_previous_objects:(sum_of_previous_objects + x_im_featurs_num),
    (x_im_featurs_num + sum_of_previous_objects):(
            x_im_next_featurs_num + x_im_featurs_num + sum_of_previous_objects)] = G_time_matrix_softmax[:, :]

    sum_of_previous_objects = x_im_featurs_num + sum_of_previous_objects

x_im = cv2.imread(image_name_list[-1])  # read input frame
pred_boxes, scores, pred_class, pred, v, predictor = get_objects(x_im)
########################## G_Space Matrix Genertaion
print("number of objects in image : ", len(pred_boxes))
dim_g_space = len(pred_boxes)
G_Space = np.zeros((dim_g_space, dim_g_space))
for row in range(dim_g_space):
    for col in range(dim_g_space):
        G_Space[row, col] = bb_intersection_over_union(pred_boxes[row], pred_boxes[col])
############################
G_Space_softmax = softmax(G_Space)
x_im_featurs_num = np.shape(pred_boxes)[0]
main_graph[sum_of_previous_objects:(sum_of_previous_objects + x_im_featurs_num),
sum_of_previous_objects:(sum_of_previous_objects + x_im_featurs_num)] = G_Space_softmax[:, :]
a, tmp_feature_maps = doit(x_im, pred_boxes, predictor)
for count in range(np.shape(tmp_feature_maps)[0]):
    node_features[sum_of_previous_objects + count, :] = tmp_feature_maps[count].cpu().view(-1)
######################################### Graph Convolution Part
######################################### Convert from numpy to networkx Graph
G = nx.from_numpy_matrix(np.matrix(main_graph))
source = []
target = []
edge_weight = []
for itm in G.edges:
    a, b = itm
    edge_weight.append(G.get_edge_data(a, b)['weight'])
    source.append(str(a))
    target.append(str(b))

square_weighted_edges = pd.DataFrame(
    {
        "source": source,
        "target": target,
        "weight": edge_weight,
    }
)

weight = edge_weight
main_graph_matrix = main_graph
dataset = 'cora'
####################################### Deep Graph Infomax train/test
# training params
batch_size = 1
nb_epochs = 10000
patience = 200
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = False
nonlinearity = 'prelu'  # special name to separate parameters

adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)  # initialization
G = nx.from_numpy_matrix(main_graph_matrix)
adj = nx.adjacency_matrix(G, weight='weight')
features = node_features
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
##### Load Model
model = DGI(ft_size, hid_units, nonlinearity)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)  # DGI optimiser

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 0.75  # 1e9
best_t = 0
#################### Train DGI for spatio-temporal Graph
for epoch in range(nb_epochs):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

    loss = b_xent(logits, lbl)

    print('Loss:', loss)

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        print("Model Saved!")
        torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break
    loss.backward()
    optimiser.step()
print("Training Finished!")
########################### Test DGI
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('best_dgi.pkl'))
model.train()
optimiser.zero_grad()

idx = np.random.permutation(nb_nodes)
shuf_fts = features[:, idx, :]

lbl_1 = torch.ones(batch_size, nb_nodes)

if torch.cuda.is_available():
    shuf_fts = shuf_fts.cuda()
    lbl = lbl_1.cuda()
start = time.time()
logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
end = time.time()
print("time " + str(end - start))
test_logits = logits[0, 0:lbl.shape[1]]
test_logits = test_logits.unsqueeze(0)
test_logits = test_logits.view(-1)
lbl = lbl.view(-1)

loss_list = []
for cnt in range(len(lbl)):
    tmp = (b_xent(test_logits[cnt], lbl[cnt]))
    loss_list.append(float(tmp.data.cpu()))
embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
embeds = embeds[0].cpu()
################### Draw TSNE for Spatio-temporal Graph
tsne = TSNE(n_components=2, random_state=42)
weighted_node_embeddings_2d = tsne.fit_transform(embeds)
alpha = 0.7

fig = px.scatter(x=weighted_node_embeddings_2d[:, 0],
                 y=weighted_node_embeddings_2d[:, 1],
                 color=loss_list,
                 color_continuous_scale=px.colors.sequential.Turbo)
fig.update_layout(
    autosize=False,
    width=1200,
    height=600, )
fig.update_traces(marker=dict(size=18))
fig.show()

print("Job Done. Have Fun!")
