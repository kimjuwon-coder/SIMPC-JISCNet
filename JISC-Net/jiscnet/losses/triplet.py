import torch
import numpy
from collections import Counter
from tslearn.clustering import TimeSeriesKMeans  # Import TimeSeriesKMeans
import torch
import slide
import torch.nn.functional as F



class PNTripletLoss(torch.nn.modules.loss._Loss):

    def __init__(self, compared_length, in_channels):
        super(PNTripletLoss, self).__init__()
        self.compared_length = compared_length
        self.in_channels = in_channels
        
        if self.compared_length is None:
            self.compared_length = numpy.inf

    def forward(self, batch, encoder, params, save_memory=False): 
        slide_num = 3
        alpha = 0.6
        total_loss = 0.0
        
        for slide_iter  in range(slide_num):   
            batch_slide = slide.slide_MTS_tensor_step(batch, alpha) #(batch, sliding 한 개수, 변수 수, 시계열 길이) #e.g.(8, n, 3, 60)
            alpha -= 0.2
            
            # do the cluster for the slided time series
            points = batch_slide.cpu().numpy()   
            num_cluster = 2
            batch_size = points.shape[0]

            anchor_list = []
            positive_list = []
            negative_list = []
            
            for b in range(batch_size):
                batch_points = points[b]  # shape: (n=21, 3, 60)
                batch_points_t = batch_points.transpose(0, 2, 1)  # shape: (21, 60, 3)

                # CPU에서 KMeans 클러스터링
                kmeans = TimeSeriesKMeans(n_clusters=num_cluster, metric="dtw", random_state=0)
                kmeans.fit(batch_points_t)
                
                cluster_label = kmeans.labels_
                num_cluster_set = Counter(cluster_label) #Counter({np.int64(0): 9, np.int64(1): 12})
                
                for pos in range(num_cluster):
                    pos_idx = numpy.where(cluster_label == pos)[0] #[ 0  1  2  3  4  5  6  7  8  9 10]
                    cluster_i = batch_points[pos_idx]  
                    cluster_i_t = cluster_i.transpose(0, 2, 1) #(9, 60, 3)
                    dist_total = kmeans.transform(cluster_i_t)
                    dist_i  = dist_total[:,pos]

                    num_pos = 50 if num_cluster_set[pos] >= 250 else int(num_cluster_set[pos] / 5 + 1)
                    sorted_indices = numpy.argsort(dist_i)
                    anchor_positive = sorted_indices[:(num_pos + 1)] #클러스터 중심에서 가장 가까운 x개를 positive로 설정 
                    anchor = torch.tensor(cluster_i[anchor_positive[0]], device=batch.device)
                    positive = torch.tensor(cluster_i[anchor_positive[1:]], device=batch.device)

                    anchor_list.append(anchor)
                    positive_list.append(positive)
                    
                    for neg in range(num_cluster):
                        if neg == pos:
                            continue
                        
                        neg_idx = numpy.where(cluster_label == neg)[0]
                        cluster_j = batch_points[neg_idx]  # (y, C, T)
                        cluster_j_t = cluster_j.transpose(0, 2, 1) #(10, 60, 3)
                        dist_total = kmeans.transform(cluster_j_t)
                        dist_j = dist_total[:, neg]
                        num_neg = 50 if num_cluster_set[neg] >= 250 else int(num_cluster_set[neg] / 5 + 1)

                        sorted_neg = numpy.argsort(dist_j)
                        neg_samples = torch.tensor(cluster_j[sorted_neg[-num_neg:]], device=batch.device)
                        negative_list.append(neg_samples)


            # Padding positives and negatives
            max_pos_len = max(p.shape[0] for p in positive_list)
            positives = torch.stack([F.pad(p, (0, 0, 0, 0, 0, max_pos_len - p.shape[0])) for p in positive_list])
            pos_mask = torch.tensor([[1] * p.shape[0] + [0] * (max_pos_len - p.shape[0]) for p in positive_list], device=batch.device)

            max_neg_len = max(n.shape[0] for n in negative_list)
            negatives = torch.stack([F.pad(n, (0, 0, 0, 0, 0, max_neg_len - n.shape[0])) for n in negative_list])
            neg_mask = torch.tensor([[1] * n.shape[0] + [0] * (max_neg_len - n.shape[0]) for n in negative_list], device=batch.device)

            anchors = torch.stack(anchor_list)  # (B, 3, 60)
            # Encode
            B, C, T = anchors.shape
            #print(f"Encoder device: {next(encoder.parameters()).device}") #수정
            
            representation_anc = encoder(anchors)
            representation_pos = encoder(positives.view(-1, C, T)).view(B, max_pos_len, -1)
            representation_neg = encoder(negatives.view(-1, C, T)).view(B, max_neg_len, -1)

            # Distance
            dist_positive = torch.norm(representation_anc.unsqueeze(1) - representation_pos, dim=2)
            dist_negative = torch.norm(representation_anc.unsqueeze(1) - representation_neg, dim=2)

            masked_dist_pos = (dist_positive * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
            masked_dist_neg = (dist_negative * neg_mask).sum(dim=1) / neg_mask.sum(dim=1)

            # Intra-cluster distance
            intra_pos = torch.cdist(representation_pos, representation_pos, p=2)
            intra_neg = torch.cdist(representation_neg, representation_neg, p=2)

            masked_intra_pos = (intra_pos * pos_mask.unsqueeze(1) * pos_mask.unsqueeze(2)).sum(dim=(1, 2)) / (pos_mask.sum(dim=1)**2 + 1e-6)
            masked_intra_neg = (intra_neg * neg_mask.unsqueeze(1) * neg_mask.unsqueeze(2)).sum(dim=(1, 2)) / (neg_mask.sum(dim=1)**2 + 1e-6)
            #torch.Size([16]) -> 배치 * 2
            
            # Final loss
            eps = 1e-6
            margin = 0.2
            slide_loss = torch.log(
                (masked_dist_pos + masked_intra_pos + masked_intra_neg) /
                (masked_dist_neg + eps) + margin
            )
            total_loss += slide_loss.mean() #배치마다 평균 
            
        return total_loss / slide_num
