#include <iostream>
#include <time.h>
#include <mpi.h>
#include "NetWork.h"
#include <unistd.h>

#include "dataLoader.h"
#include "Utils.h"
#include <fstream>
#include <string>
/** indicate 0 ~ 9 */
#define NUM_NET_OUT 10


#define TRAIN_IMAGES_URL "../data/train-images.idx3-ubyte"
#define TRAIN_LABELS_URL "../data/train-labels.idx1-ubyte"

#define TEST_IMANGES_URL "../data/t10k-images.idx3-ubyte"
#define TEST_LABELS_URL  "../data/t10k-labels.idx1-ubyte"

#define CONFIG_FILE      "config.txt"

static int SIZE = 0;
static int NUM_HIDDEN = 0;
static double NET_LEARNING_RATE = 0;

void getConfig(){
    ifstream f;
    f.open(CONFIG_FILE , ios::in);
    sleep(0.1);
    if(f){
       
        f >> NET_LEARNING_RATE >> NUM_HIDDEN >> SIZE;
    }
    else cout << "open fail";
}

/*void showNumber(unsigned char pic[], int width, int height) {
    int idx = 0;
    for (int i=0; i < height; i++) {
        for (int j = 0; j < width; j++ ) {

            if (pic[idx++]) {
                cout << "1";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }
}*/

inline void preProcessInputData(const unsigned char src[], double out[], int size) {
    for (int i = 0; i < size; i++) {
        out[i] = (src[i] >= 128) ? 1.0 : 0.0;
    }
}

/*inline void preProcessInputDataWithNoise(const unsigned char src[], double out[], int size) {
    for (int i = 0; i < size; i++) {
        out[i] = ((src[i] >= 128) ? 1.0 : 0.0) + RandFloat() * 0.1;
    }
}*/

/*inline void preProcessInputData(const unsigned char src[],int size, std::vector<int>& indexs) {
    for (int i = 0; i < size; i++) {
        if (src[i] >= 128) {
            indexs.push_back(i);
        }
    }
}*/

double trainEpoch(dataLoader& src, NetWork& bpnn, int imageSize, int numImages) {
    //for mpi
    
    int task_count = 0;
    int rank = 0;
    int tag = 0;
    MPI_Status status;
    //for train
    double net_target[NUM_NET_OUT];
    char* temp = new char[imageSize];
    double* net_train = new double[imageSize];
    
    //get mpi message
    MPI_Comm_size(MPI_COMM_WORLD, &task_count);  //get num of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);        //get current rank number
    --task_count;                                //the num of ranks used for training
    double comun_time = 0.0;
    for (int i = 0; i < numImages;) {
    	
        int row1 = bpnn.mNeuronLayers[0]->mNumNeurons;
        int row2 = bpnn.mNeuronLayers[1]->mNumNeurons;
        int col1 = bpnn.mNeuronLayers[0]->mNumInputsPerNeuron + 1;
        int col2 = bpnn.mNeuronLayers[1]->mNumInputsPerNeuron + 1;
        double weights1[row1][col1];
        double weights2[row2][col2];

        double new_weights1[row1][col1];
        double new_weights2[row2][col2];
        
        if(rank != 0){
            int sample_num = 0;
            if(i + task_count * SIZE > numImages){
                sample_num = (numImages - i) / task_count;
                if(rank <= ((numImages - i) % task_count))
                    sample_num++;
            }
            else{
                sample_num = SIZE;
            }
            for(int loop = 0; loop < sample_num; loop++){
                int label = 0;
                memset(net_target, 0, NUM_NET_OUT * sizeof(double));
                if (src.read(&label, temp, i + ((rank-1) * sample_num) + loop)) {
                    net_target[label] = 1.0;
                    preProcessInputData((unsigned char*)temp, net_train, imageSize);
                    bpnn.training(net_train, net_target);
                }
                else {
                    cout << "读取训练数据失败" << endl;
                    break;
                }
            }
        }
        if(rank != 0){
            for(int loop = 0; loop < row1; loop++){
                for(int loop1 = 0; loop1 < col1; loop1++)
                    weights1[loop][loop1] = bpnn.mNeuronLayers[0]->mWeights[loop][loop1];
            }
            for(int loop = 0; loop < row2; loop++){
                for(int loop1 = 0; loop1 < col2; loop1++)
                    weights2[loop][loop1] = bpnn.mNeuronLayers[1]->mWeights[loop][loop1];
            }
            for(int loop = 0; loop < row1; loop++){
                MPI_Send(weights1[loop], col1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);    
            }
            MPI_Send(weights2, row2*col2, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
            
        MPI_Barrier(MPI_COMM_WORLD);
            
        if(rank == 0){//father rank
        	double cur_time = MPI_Wtime();
            for(int loop = 0; loop < row1; loop++){
                //new_weights1[loop] = new double[col1];
                for(int loop1 = 0; loop1 < col1; loop1++)
                    new_weights1[loop][loop1] = 0;
            }
            for(int loop = 0; loop < row2; loop++){
                //new_weights2[loop] = new double[col2];
                for(int loop1 = 0; loop1 < col2; loop1++)
                    new_weights2[loop][loop1] = 0;
            }

            for(int j = 1; j <= task_count; j++){//recv and calculate the new weights
               	//double recv_time = MPI_Wtime();
                for(int loop = 0; loop < row1; loop++)
                    MPI_Recv(weights1[loop], row1*col1, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
                //MPI_Recv(weights1, row1*col1, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
                MPI_Recv(weights2, row2*col2, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);
                //recv_time = MPI_Wtime() - cur_time;
                for(int loop = 0; loop < row1; loop++){
                    for(int loop1 = 0; loop1 < col1; loop1++)
                        new_weights1[loop][loop1] += weights1[loop][loop1];
                }
                for(int loop = 0; loop < row2; loop++){
                    for(int loop1 = 0; loop1 < col2; loop1++)
                        new_weights2[loop][loop1] += weights2[loop][loop1];
                }
                //cur_time += recv_time;
            }

            for(int loop = 0; loop < row1; loop++){
                for(int loop1 = 0; loop1 < col1; loop1++)
                    new_weights1[loop][loop1] /= task_count;
            }
            for(int loop = 0; loop < row2; loop++){
                for(int loop1 = 0; loop1 < col2; loop1++)
                    new_weights2[loop][loop1] /= task_count;
            }
        	
            for(int j = 1; j <= task_count; j++){
            	//double send_time = MPI_Wtime();
                for(int loop = 0; loop < row1; loop++)
                    MPI_Send(new_weights1[loop], col1, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
                MPI_Send(new_weights2, row2*col2, MPI_DOUBLE, j, tag, MPI_COMM_WORLD);
                //send_time = MPI_Wtime() - send_time;
                //cur_time += send_time;
            }
            for(int loop = 0; loop < row1; loop++){
                for(int loop1 = 0; loop1 < col1; loop1++)
                    bpnn.mNeuronLayers[0]->mWeights[loop][loop1] = new_weights1[loop][loop1];
            }
            for(int loop = 0; loop < row2; loop++){
                for(int loop1 = 0; loop1 < col2; loop1++)
                    bpnn.mNeuronLayers[1]->mWeights[loop][loop1] = new_weights2[loop][loop1];
            }
            cout << "已学习：" << i << "\r";
            cur_time = MPI_Wtime() - cur_time;
            comun_time += cur_time;
        }
        if(rank !=0){
                //get new weights
            for(int loop = 0; loop < row1; loop++)
            MPI_Recv(weights1, col1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
            
            MPI_Recv(weights2, row2*col2, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
            for(int loop = 0; loop < row1; loop++){
                for(int loop1 = 0; loop1 < col1; loop1++)
                    bpnn.mNeuronLayers[0]->mWeights[loop][loop1] = weights1[loop][loop1];
            }
            for(int loop = 0; loop < row2; loop++){
                for(int loop1 = 0; loop1 < col2; loop1++)
                    bpnn.mNeuronLayers[1]->mWeights[loop][loop1] = weights2[loop][loop1];
            }
               
        }
        MPI_Barrier(MPI_COMM_WORLD);
        i += task_count*SIZE;
        
    }
    if(rank == 0)
    	cout << " comun_time=" << comun_time << endl;
    // cout << "the error is:" << bpnn.getError() << " after training " << endl;

    delete []net_train;
    delete []temp;

    return bpnn.getError();

}

int testRecognition(dataLoader& testData, NetWork& bpnn, int imageSize, int numImages) {
    int ok_cnt = 0;
    double* net_out = NULL;
    char* temp = new char[imageSize];
    double* net_test = new double[imageSize];
    for (int i = 0; i < numImages; i++) {
        int label = 0;

        if (testData.read(&label, temp, i)) {			
            preProcessInputData((unsigned char*)temp, net_test, imageSize);
            bpnn.process(net_test, &net_out);

            int idx = -1;
            double max_value = -99999;
            for (int i = 0; i < NUM_NET_OUT; i++) {
                if (net_out[i] > max_value) {
                    max_value = net_out[i];
                    idx = i;
                }
            }
            if(i%100 == 0)
            	cout << "result:" << idx << " label:" << label <<endl; 
            if (idx == label) {
                ok_cnt++;
            }

        }
        else {
            cout << "read test data failed" << endl;
            break;
        }
    }
    delete []net_test;
    delete []temp;
    return ok_cnt;
}


int main(int argc, char* argv[]) {
    
    int ret = MPI_Init(&argc, &argv);
    int rank = 0;
    if(MPI_SUCCESS != ret) {
        printf("start mpi fail\n");
        MPI_Abort(MPI_COMM_WORLD, ret);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    getConfig();
    
    dataLoader src;
    dataLoader testData;
    NetWork* bpnn = NULL;
    srand((int)time(0));
    
    if (src.openImageFile(TRAIN_IMAGES_URL) && src.openLabelFile(TRAIN_LABELS_URL)) {
        int imageSize = src.imageLength();
        int numImages = 60000;
        int epochMax = 1;
        double expectErr = 0.1;
        bpnn = new NetWork(imageSize, NET_LEARNING_RATE);
        // 加入隐藏层
        bpnn->addNeuronLayer(NUM_HIDDEN);
        // 加入输出层
        bpnn->addNeuronLayer(NUM_NET_OUT);
        uint64_t st = MPI_Wtime();
        
            
        for (int i = 0; i < epochMax; i++) {
        	if(rank == 0){
        	cout << "开始进行number "<<i+1<<" times 训练：" << endl;
        	}
            double err = trainEpoch(src, *bpnn, imageSize, numImages);
            src.reset();
        }
        
        if(rank == 0){
            cout << "训练结束，花费时间: " << (MPI_Wtime() - st) << "秒" << endl;
        st = MPI_Wtime();
        
        if (testData.openImageFile(TEST_IMANGES_URL) && testData.openLabelFile(TEST_LABELS_URL)) {
            imageSize = testData.imageLength();
            numImages = testData.numImage();
            
            cout << "开始进行测试：" << endl;

            int ok_cnt = testRecognition(testData, *bpnn, imageSize, numImages);

            cout << "测试结束，花费时间："
                << (MPI_Wtime() - st)<< "秒, " 
                <<  "成功比例: " << ok_cnt/(double)numImages*100 << "%" << endl;
        }
        else {
            cout << "打开测试文件失败" << endl;
        }
        }
    }
    else {
        cout << "open train image file failed" << endl;
    }

    if (bpnn) {
        delete bpnn;
    }
    if(rank == 0)
        getchar();
    MPI_Finalize();
    return 0;
}