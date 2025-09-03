import math
import numpy
import numpy as np
import torch
import sklearn
import joblib
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.svm import SVC
import timeit
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
from sklearn.metrics import f1_score
import pickle
import matplotlib.pyplot as plt
import utils
import losses
import networks
import slide

class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):

    def __init__(self, compared_length,
                 batch_size, epochs, lr,
                 encoder, params, in_channels, cuda=False, gpu=0, shapelet_num = 6):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr 
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.loss = losses.triplet.PNTripletLoss(
            compared_length, in_channels
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.shapelet_num = shapelet_num

    def save_shapelet(self, prefix_file, shapelet, shapelet_label, utility_sort_index, average, f1_macro, f1_micro, test, test_label):
         # Save shapelet data as pickle files
        with open(prefix_file + "_shapelet.pkl", 'wb') as f:
            pickle.dump(shapelet, f)
        print(f"Shapelet data saved to {prefix_file}_shapelet.pkl")
        
        with open(prefix_file + "_shapelet_label.pkl", 'wb') as f:
            pickle.dump(shapelet_label, f)
        print(f"Shapelet labels saved to {prefix_file}_shapelet_label.pkl")
        
        with open(prefix_file + "_utility_sort_index.pkl", 'wb') as f:
            pickle.dump(utility_sort_index, f)
        print(f"Utility sort index saved to {prefix_file}_utility_sort_index.pkl")
        
        with open(prefix_file + "_test.pkl", 'wb') as f:
            pickle.dump(test, f)
        print(f"Shapelet data saved to {prefix_file}_test.pkl")
        
        with open(prefix_file + "_test_label.pkl", 'wb') as f:
            pickle.dump(test_label, f)
        print(f"Shapelet data saved to {prefix_file}_test_label.pkl")
        
        # Save evaluation metrics to a text file
        with open(prefix_file + "_evaluation_metrics.txt", 'w') as f:
            f.write(f"Average: {average:.5f}\n")
            f.write(f"F1-macro: {f1_macro:.5f}\n")
            f.write(f"F1-micro: {f1_micro:.5f}\n")
        print(f"Evaluation metrics saved to {prefix_file}_evaluation_metrics.txt")
        
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )


    def load_shapelet(self, prefix_file):
        '''
        load the shapelet and its dimension from disk
        '''
        # save shapelet
        fo_shapelet = prefix_file+"shapelet.txt"
        with open(fo_shapelet, "r") as fo_shapelet:
            shapelet = []
            for line in fo_shapelet:
                shapelet.append(line)
        fo_shapelet.close()

        # save shapelet dimension
        fo_shapelet_dim = open(prefix_file+"shapelet_dim.txt", "r")
        shapelet_dim = numpy.loadtxt(fo_shapelet_dim)
        fo_shapelet_dim.close()

        return shapelet, shapelet_dim

    def save_encoder(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        joblib.dump(
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def fit_svm_linear(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an svm linear
        classifier.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        self.classifier = SVC(kernel='linear', probability=True, gamma='auto')
        self.classifier.fit(features, y)

        return self.classifier

    def fit_encoder(self, X, y=None, save_memory=True, verbose=True, prefix_file='/JISC-Net/jiscnet/result/'):
        """
        Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = utils.Dataset(X) #(3, 100) 이 743개
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Encoder training 
        losses = []
        best_loss = float('inf')  # 가장 낮은 loss 초기값
        no_improve_count = 0      # 개선되지 않은 epoch 수
        
        for i in range(self.epochs):
            epoch_start = timeit.default_timer()
            epoch_loss = 0
            
            for batch_num, batch in enumerate(train_generator):  
                batch_start = timeit.default_timer()
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                #(1) 기존 gradient 초기화
                self.optimizer.zero_grad() 
                #(2) 손실 함수 계산                    
                loss = self.loss(
                   batch, self.encoder, self.params, save_memory=save_memory
                ) 
                #(3) 역전파로 gradient 계산
                loss.backward()
                #(4) gradient를 이용해 encoder 파라미터 업데이트
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_end = timeit.default_timer()
                print("batch time: ", (batch_end- batch_start)/60)
                
                    
            epoch_end = timeit.default_timer()
            avg_loss = epoch_loss / (batch_num + 1)  # 평균 loss
            losses.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(prefix_file)  # 가장 낮은 loss일 때만 저장
                print(f"✅ Epoch {i}: Loss improved to {avg_loss:.6f}. Model saved.")
            else:
                no_improve_count += 1
                print(f"⚠️ Epoch {i}: Loss {avg_loss:.6f} did not improve (best: {best_loss:.6f})")


            if no_improve_count >= 20:
                print(f"🛑 Early stopping: No improvement for {no_improve_count} consecutive epochs.")
                break
            print("epoch time: ", (epoch_end- epoch_start)/60)

        # === 시각화 ===
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
        return self.encoder

    def fit(self, X, y, test, test_labels, prefix_file, cluster_num, save_memory=True, verbose=True):
        """
        Trains sequentially the encoder unsupervisedly and then the classifier
        using the given labels over the learned features.

        @param X Training set.
        @param y Training labels.
        @param test testing set.
        @param test_labels testing labels.
        @param prefix_file prefix path.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        print('def fit')
        # Fitting encoder
        encoder_start = timeit.default_timer()
        self.encoder = self.fit_encoder(
                                        X, y=y, save_memory=save_memory, verbose=verbose
                                        , prefix_file=prefix_file)
        encoder_end = timeit.default_timer()
        print("encode time: ", (encoder_end- encoder_start)/60)

        # shapelet discovery
        discovery_start = timeit.default_timer()
        shapelet, shapelet_label, utility_sort_index = self.shapelet_discovery(X, y, cluster_num, batch_size=50) #((cluster_num, 3, 60))
        print('shapelet_label num: ', len(shapelet_label))
        discovery_end = timeit.default_timer()
        print("discovery time: ", (discovery_end- discovery_start)/60)

        # shapelet transformation - 각 인스턴스와 각 shapelet의 최소 dtw 거리
        transformation_start = timeit.default_timer()
        features = self.shapelet_transformation(X, shapelet, utility_sort_index) #(인스턴스 수, final_shapelet_num)
        transformation_end = timeit.default_timer()
        print("transformation time: ", (transformation_end - transformation_start)/60)

        # SVM classifier training
        classification_start = timeit.default_timer()
        self.classifier = self.fit_svm_linear(features, y)
        classification_end = timeit.default_timer()
        print("classification time: ", (classification_end - classification_start)/60)
        
        average, f1_macro, f1_micro = self.score(test, test_labels, shapelet, utility_sort_index)
        self.save_shapelet(prefix_file, shapelet, shapelet_label, utility_sort_index, average, f1_macro, f1_micro, test, test_labels)
        return self
    
    
    def shapelet_discovery(self, X, train_labels, cluster_num, batch_size = 50):
        '''
        slide raw time series as candidates
        encode candidates
        cluster new representations
        select the one nearest to centroid
        trace back original candidates as shapelet
        ''' 

        slide_num = 3
        alpha = 0.6
        X_slide_num = []
        
        representation_all = numpy.empty((0, 0)) #초기화 
        representation_class_label = numpy.empty((0,))

        for m in range(slide_num):
            # slide the raw time series and the corresponding class and variate label
            X_slide, candidates_class_label = slide.slide_MTS_dim_step(X, train_labels, alpha)
            #X_slide: (7833, 3, 60)
            X_slide_num.append(numpy.shape(X_slide)[0]) #sliding한것의 instance 수 #[7833, 11563, 15666]
            alpha = alpha - 0.2
            count = 0

            test = utils.Dataset(X_slide) #실제 test dataset은 아니고 학습된 encoder에 train dataset을 넣는 것 
            test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size) # (batch_size=50, 변수 수, final_output_size)

            self.encoder = self.encoder.eval()

            # encode slide TS
            with torch.no_grad():
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu) # (batch_size, 변수 수, final_output_size)

                    batch = self.encoder(batch).cpu().numpy() # (batch_size, final_output_size)

                    if count == 0:
                        representation = batch
                    else:
                        representation = numpy.concatenate((representation, batch), axis=0)
                    count += 1
            self.encoder = self.encoder.train()
            
            # concatenate the new representation from different slides
            if m == 0 :
                representation_all = representation #삼차원 
                representation_class_label = candidates_class_label #일차원 
            else:
                representation_all = numpy.concatenate((representation_all, representation), axis = 0)
                representation_class_label = numpy.concatenate((representation_class_label, candidates_class_label), axis=0)
               
        #representation_all: (7833+11563+15666, 16)
        #representation_class_label: (7833+11563+15666,)
            

        # cluster all the new representations
        num_cluster = cluster_num #일단 지금은 data의 class수로 고정해 놓음 6
        kmeans = KMeans(n_clusters=num_cluster, random_state=0) 
        kmeans.fit(representation_all)

        # init candidate as list
        candidate = []
        candidate_label_list = []
        # candidate_dim = numpy.zeros(num_cluster) # shape: (cluster_num,)
        # two parts of utility function
        candidate_cluster_size = []
        candidate_first_representation = []
        utility = []

        # select the nearest to the centroid
        for i in range(num_cluster):
            candidate_i_size = representation_all[kmeans.labels_==i][:,0].size
            candidate_cluster_size.append(candidate_i_size) #7833+11563+15666개 중에서 label이 0, 1, ...인 것의 갯수를 list로 
            class_label_cluster_i = list()
            dist = math.inf
            
            for j in range(candidate_i_size): 
                #(representation_all[kmeans.labels_==i][j]) == (kmeans의 i label에 j 번째 index (16,))
                target = representation_all[kmeans.labels_==i][j]  # 현재 벡터
                match = np.where(np.all(np.isclose(representation_all, target, atol=1e-6), axis=1))[0] #해당 벡터와 동일한 모든 벡터들의 행 인덱스 반환. 약간의 오차 허용 
                
                dist_tmp = numpy.linalg.norm(target - kmeans.cluster_centers_[i]) #속한 클러스터의 중심과 해당 point와의 유클리드 거리
                for k in range(match.shape[0]):
                    class_label_cluster_i.append(representation_class_label[match[k]]) #동일한 벡터들의 라벨 분포를 분석해서, 이 shapelet이 특정 클래스와 얼마나 잘 연결되어 있는지 파악을 위함. 
                    #[0, 0, 1, 2, ..]
                if dist_tmp < dist:
                    dist = dist_tmp
                    candidate_label = representation_class_label[match[0]]
                    # record the first representation
                    tmp_candidate_first_representation = target
                    # trace back the original candidates
                    nearest = numpy.where(representation_all == (target)) #nearest[0][0]: representation_all 에서 target의 인덱스 
                    
                    sum_X_slide_num = 0
                    for k in range(slide_num):
                        sum_X_slide_num += X_slide_num[k] #슬라이드 별 누적 개수
                        if (nearest[0][0] < sum_X_slide_num):
                            index_slide = nearest[0][0] - sum_X_slide_num + X_slide_num[k] #어느 슬라이드의 몇 번째 index 의 것인지 추적 
                            X_slide_disc = slide.slide_MTS_dim(X, (0.6-k*0.2))
                            candidate_tmp = X_slide_disc[index_slide] # 해당 조각을 가져옴 #(3, 100*alpha)
                            #candidate_dim[i] = index_slide % numpy.shape(X)[1]
                            break
                        
            #class_label_top = (counter.most_common(1)[0][1] / len(class_label_cluster_i)) #cluster i의 shapelet 들과 유사한 shapelet 들이 가장 많이 있는 class(train data의 label), 몇 프로가 해당 class에 속하는지
            #이 i 번째 cluster에서 뽑은 shapelet이 train 의 class 중 어느 정도 이상 관련이 있어서 진행.
            counter = Counter(class_label_cluster_i)
            if ((counter.get(candidate_label, 0) / len(class_label_cluster_i)) < (1/numpy.unique(train_labels).shape[0])): #shapelet 후보의 갯수가 해당 cluster에서 1/(전체 클래스 수) 보다 커야 인정
                continue
            
            # append the first representation
            candidate_first_representation.append(tmp_candidate_first_representation) #해당 벡터가 이 cluster를 대표하는 패턴으로 저장됨 
            # list append method
            candidate.append(candidate_tmp)
            candidate_label_list.append(candidate_label)

        # utility
        for i in range(len(candidate_first_representation)):
            ed_dist_sum = 0
            for j in range(len(candidate_first_representation)):
                ed_dist_sum += numpy.linalg.norm(candidate_first_representation[i] - candidate_first_representation[j])
            utility.append(candidate_cluster_size[i] * ed_dist_sum)
            # 클러스터의 크기와 다른 shapelet들 간의 거리 곱 -> 클수록 우선순위 
            
        # sort utility namely candidate
        utility_sort_index = numpy.argsort(-numpy.array(utility))

        return candidate, candidate_label_list, utility_sort_index #candidate, candidate_dim, utility_sort_index

    def shapelet_transformation(self, X, candidate, utility_sort_index):
        '''
        transform the original multivariate time series into the new one vector data space
        transformed date label the same with original label
        '''
        # init transformed data with list
        feature = []

        # transform original time series
        print('shapelet_transformation: ')
        #print('np.shape(X): ', np.shape(X)) #(373, 3, 100)
        
        for i in range(np.shape(X)[0]): #373
            for j in range(len(candidate)):
                dist = math.inf
                candidate_tmp = np.asarray(candidate[utility_sort_index[j]]) #(3, l=60)
    
                # Loop over the time series and calculate DTW distance
                for k in range(np.shape(X)[2] - np.shape(candidate_tmp)[1] + 1): #100 - 60 + 1
                    # Extract a segment from the time series
                    series_segment = X[i, :, k : k + np.shape(candidate_tmp)[1]]
                    
                    # Calculate DTW distance between the candidate shapelet and the time series segment
                    distance, _ = fastdtw(series_segment.T, candidate_tmp.T, dist=euclidean)
                    
                    # Update the minimum distance
                    if distance < dist:
                        dist = distance 
    
                # Append the minimum DTW distance for this candidate shapelet
                feature.append(dist) #(shapelet 수)
    
        # Convert the feature list to a numpy array and reshape it to the correct format
        feature = np.asarray(feature)
        feature = feature.reshape(np.shape(X)[0], len(candidate)) #(373, shapelet 수): 각 인스턴스 별 모든 shapelets 과의 거리 계산
    
        return feature
   
   
    def score(self, X, y, shapelet, utility_sort_index):
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.shapelet_transformation(X, shapelet, utility_sort_index)
        # Step 2: 모델의 정확도 계산
        accuracy = self.classifier.score(features, y)

        # Step 3: F1-macro score 계산
        y_pred = self.classifier.predict(features)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_micro = f1_score(y, y_pred, average='micro')

        # Step 4: 정확도와 F1-macro score를 함께 반환
        return accuracy, f1_macro, f1_micro

class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """
    Wraps a causal CNN encoder of time series as a PyTorch module and a
    SVM classifier on top of its computed representations in a scikit-learn
    class.

    @param compared_length Length of the compared positive and negative samples
           in the loss. Ignored if None, or if the time series in the training
           set have unequal lengths.
    @param nb_random_samples Number of randomly chosen intervals to select the
           final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    @param batch_size Batch size used during the training of the encoder.
    @param epochs Number of epochs to run during the training of the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the
           number of samples is high enough, performs a hyperparameter search
           to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic
           for the training of the representations, based on the final
           score. Representations are still learned unsupervisedly in this
           case. If the number of samples per class is no more than 10,
           disables this heuristic. If not None, accepts an integer
           representing the patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """
    def __init__(self, compared_length=10, batch_size=1, epochs=100, lr=0.001,
                 channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0, shapelet_num = 6):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, batch_size,
            epochs, lr,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, cuda, gpu, shapelet_num
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        """
        Outputs the representations associated to the input by the encoder,
        from the start of the time series to each time step (i.e., the
        evolution of the representations of the input time series with
        repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), wich
        ensures that its output at time step i only depends on time step i and
        previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                # First applies the causal CNN
                output_causal_cnn = causal_cnn(batch)
                after_pool = torch.empty(
                    output_causal_cnn.size(), dtype=torch.double
                )
                if self.cuda:
                    after_pool = after_pool.cuda(self.gpu)
                after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                # Then for each time step, computes the output of the max
                # pooling layer
                for i in range(1, length):
                    after_pool[:, :, i] = torch.max(
                        torch.cat([
                            after_pool[:, :, i - 1: i],
                            output_causal_cnn[:, :, i: i+1]
                         ], dim=2),
                        dim=2
                    )[0]
                features[
                    count * batch_size: (count + 1) * batch_size, :, :
                ] = torch.transpose(linear(
                    torch.transpose(after_pool, 1, 2)
                ), 1, 2)
                count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu,
            'shapelet_num': self.shapelet_num
        }

    def set_params(self, compared_length, batch_size, epochs, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu, shapelet_num):
        self.__init__(
            compared_length, batch_size, epochs, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu, shapelet_num
        )
        return self


