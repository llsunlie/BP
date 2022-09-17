/**
 * @file BP神经网络2000_30_15.cpp
 * @author 201905555514 刘世豪
 * @brief 三层BP神经网络实现人脸识别
 * @version 0.1
 * @date 2022-05-06
 * 
 * @copyright Copyright (c) 2022 liushihao
 * 
 */
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <cstring>
#include <cfloat>
using namespace std;
#define in   first
#define out  second

// BP 神经网络层数为 3 (不可更改!!!)
#define LAYERS  3
// 输入层神经元个数
#define INPUT_NEURONS_NUMBER    2000
// 中间层神经元个数
#define MIDDLE_NEURONS_NUMBER   30
// 输出层神经元个数
#define OUTPUT_NEURONS_NUMBER   15
// 训练的最大迭代数
#define MAXIMUM_TRAINING_ROUNDS 100000000
// 学习效率
#define LEARNING_EFFICIENCY     0.5
// 平均误差限制
#define AVERAGE_ERROR_LIMIT     0.005
// 图片局部的左上右下的坐标
#define UPPER_LEFT_CORNER_X     0
#define UPPER_LEFT_CORNER_Y     0
#define LOWER_RIGHT_CORNER_X    79
#define LOWER_RIGHT_CORNER_Y    99
// 参数文件
#define PATAMETER_FILEPATH      "parameter2000_0.txt"
// 每隔多少次迭代保存一次参数
#define SAVE_PARAMETER_ROUNDS   100

/**
 * @brief BMP类（存放BMP相关信息）
 * @date 2022-05-06
 */
class BMP{
public:
    // 图像类型
    unsigned short bfType;

    // 图像头结构体
    struct BitmapFileHeader{
        // unsigned short bfType;        // 19778，必须是BM字符串，对应的十六进制为0x4d42,十进制为19778，否则不是bmp格式文件
        unsigned int   bfSize;        // 文件大小 以字节为单位(2-5字节)
        unsigned short bfReserved1;   // 保留，必须设置为0 (6-7字节)
        unsigned short bfReserved2;   // 保留，必须设置为0 (8-9字节)
        unsigned int   bfOffBits;     // 从文件头到像素数据的偏移  (10-13字节)
    } bitmapFileHeader;

    // 图像信息头结构体
    struct BitmapInfoHeader{
        unsigned int    biSize;          // 此结构体的大小 (14-17字节)
        unsigned int    biWidth;         // 图像的宽  (18-21字节)
        unsigned int    biHeight;        // 图像的高  (22-25字节)
        unsigned short  biPlanes;        // 表示bmp图片的平面属，显然显示器只有一个平面，所以恒等于1 (26-27字节)
        unsigned short  biBitCount;      // 一像素所占的位数，一般为24   (28-29字节)
        unsigned int    biCompression;   // 说明图象数据压缩的类型，0为不压缩。 (30-33字节)
        unsigned int    biSizeImage;     // 像素数据所占大小, 这个值应该等于上面文件头结构中bfSize-bfOffBits (34-37字节)
        unsigned int    biXPelsPerMeter; // 说明水平分辨率，用象素/米表示。一般为0 (38-41字节)
        unsigned int    biYPelsPerMeter; // 说明垂直分辨率，用象素/米表示。一般为0 (42-45字节)
        unsigned int    biClrUsed;       // 说明位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）。 (46-49字节)
        unsigned int    biClrImportant;  // 说明对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要。(50-53字节)
    } bitmapInfoHeader;

    // 24位图像素信息结构体(调色板)
    struct ColorInfo {
        unsigned char rgbBlue;   //该颜色的蓝色分量  (值范围为0-255)
        unsigned char rgbGreen;  //该颜色的绿色分量  (值范围为0-255)
        unsigned char rgbRed;    //该颜色的红色分量  (值范围为0-255)
        unsigned char rgbReserved;// 保留，必须为0
    } ;
    vector<ColorInfo> colorInfo;

    // 图像像素信息
    vector<unsigned char> pixelInfo;

    /**
     * @brief 通过文件路径载入BMP
     * @date 2022-05-06
     * @param filePath 文件路径
     */
    BMP(const char *filePath){
        FILE *fp = fopen(filePath, "rb");
        if(fp == NULL){
            printf("打开图片失败!!\n");
            return;
        }
        // 读取图像类型
        fread(&bfType, 2, 1, fp);
        // 读取图像头结构体
        fread(&bitmapFileHeader, sizeof(bitmapFileHeader), 1, fp);
        // 读取图像信息头结构
        fread(&bitmapInfoHeader, sizeof(bitmapInfoHeader), 1, fp);
        // 读取调色板
        colorInfo.resize( 1 << bitmapInfoHeader.biBitCount );
        fread(&colorInfo[0], sizeof(ColorInfo), colorInfo.size(), fp);
        // 读取图像像素信息
        pixelInfo.resize( bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight );
        fread(&pixelInfo[0], 1, pixelInfo.size(), fp);
        fclose(fp);
    }

    /**
     * @brief 打印BMP文件头
     * @date 2022-05-06
     */
    void showBitmapFileHeader(){
        printf("BitmapFileHeader{\n");
        printf("\tbfType:\t\t%d\n",     bfType);
        printf("\tbfSize:\t\t%d\n",     bitmapFileHeader.bfSize);
        printf("\tbfReserved1:\t%d\n",  bitmapFileHeader.bfReserved1);
        printf("\tbfReserved2:\t%d\n",  bitmapFileHeader.bfReserved2);
        printf("\tbfOffBits:\t%d\n",    bitmapFileHeader.bfOffBits);
        printf("}\n");
    }

    /**
     * @brief 打印BMP信息头
     * @date 2022-05-06
     */
    void showBitmapInfoHeader(){
        printf("BitmapInfoHeader{\n");
        printf("\tbiSize:\t\t%d\n",         bitmapInfoHeader.biSize);   
        printf("\tbiWidth:\t%d\n",          bitmapInfoHeader.biWidth);
        printf("\tbiHeight:\t%d\n",         bitmapInfoHeader.biHeight);
        printf("\tbiPlanes:\t%d\n",         bitmapInfoHeader.biPlanes);
        printf("\tbiBitCount:\t%d\n",       bitmapInfoHeader.biBitCount);
        printf("\tbiCompression:\t%d\n",    bitmapInfoHeader.biCompression);
        printf("\tbiSizeImage:\t%d\n",      bitmapInfoHeader.biSizeImage);
        printf("\tbiXPelsPerMeter:%d\n",    bitmapInfoHeader.biXPelsPerMeter);
        printf("\tbiYPelsPerMeter:%d\n",    bitmapInfoHeader.biYPelsPerMeter);
        printf("\tbiClrUsed:\t%d\n",        bitmapInfoHeader.biClrUsed);
        printf("\tbiClrImportant:\t%d\n",   bitmapInfoHeader.biClrImportant);
        printf("}\n");
    }

    /**
     * @brief 打印BMP颜色盘
     * @date 2022-05-06
     */
    void showColorInfo(){
        printf("ColorInfo{\n");
        for(int i = 0 ; i < (1 << bitmapInfoHeader.biBitCount); i++){
            printf("\t%0.2X %0.2X %0.2X %0.2X\n", colorInfo[i].rgbBlue, colorInfo[i].rgbGreen, colorInfo[i].rgbRed, colorInfo[i].rgbReserved);
        }
        printf("}\n");
    }

    /**
     * @brief 打印BMP像素信息
     * @date 2022-05-06
     */
    void showPixelInfo(){
        printf("PixelInfo{\n");
        for(int i = 0; i < bitmapInfoHeader.biWidth * bitmapInfoHeader.biHeight; i++){
            printf("%0.2X ", pixelInfo[i]);
        }
        printf("}\n");
    }
};

/**
 * @brief BP神经网络类
 * @date 2022-05-06
 */
class BPNeuralNetwork{
public:
    // 图片对应的文件路径
    vector<const char *> bmpFilePathList;
    // 需要训练/判断的图片集合
    vector<BMP> bmpList;
    // 图片的所有者编号
    vector<int> bmpOwnerList;
    // 各层节点的输入输出
    vector< vector< pair< double, double > > > layer;
    // 各层之间的权重
    vector< vector< vector< double > > > weight;
    // 各层之间的权值的变化量
    vector< vector< vector< double > > > weightVariation;
    // 中间层和输出层的阈值
    vector< vector< double > > thresholds;
    // 中间层和输出层的阈值的变化量
    vector< vector< double > > thresholdsVariation;
    // 中间层和输出层的误差项
    vector< vector< double > > error;
    // 输出层的各节点期望值
    vector< double > expect;

    /**
     * @brief 导入多张BMP至BP网络
     * @date 2022-05-06
     * @param filePathList 文件路径集合
     */
    BPNeuralNetwork(const vector<const char *> &filePathList){
        for(auto &filePath : filePathList){
            BMP bmp(filePath);
            bmpList.push_back(bmp);
            bmpOwnerList.push_back(calId(filePath));
            bmpFilePathList.push_back(filePath);
        } 
    }

    /**
     * @brief 计算该BMP对应人的编号
     * @date 2022-05-06
     * @param filePath 文件路径
     * @return int 返回编号(编号从 0 开始)
     */
    int calId(const char *filePath){
        return (filePath[13] - '0') * 10 + (filePath[14] - '0') - 1;
    }

    /**
     * @brief sigmoid函数
     * @date 2022-05-06
     * @param x 
     * @return double 
     */
    double f(double x){
        return 1.0 / ( 1.0 + exp(-x) );
    }

    /**
     * @brief sigmoid函数的导数
     * @date 2022-05-06
     * @param x 
     * @return double 
     */
    double df(double x){
        return f(x) * (1.0 - f(x));
    }

    /**
     * @brief 初始化各个变量空间
     * @date 2022-05-06
     */
    void initVariableSpace(){
        // 初始化存放各层节点的 空间
        layer.resize(LAYERS);
        layer[0].resize(INPUT_NEURONS_NUMBER);
        layer[1].resize(MIDDLE_NEURONS_NUMBER);
        layer[2].resize(OUTPUT_NEURONS_NUMBER);
        // 初始化存放权重的 空间
        weight.resize(LAYERS - 1);
        weight[0].resize(INPUT_NEURONS_NUMBER);
        for(auto &tmp : weight[0]) tmp.resize(MIDDLE_NEURONS_NUMBER);
        weight[1].resize(MIDDLE_NEURONS_NUMBER);
        for(auto &tmp : weight[1]) tmp.resize(OUTPUT_NEURONS_NUMBER);
        // 初始化存放阈值变化量的 空间
        weightVariation.resize(LAYERS - 1);
        weightVariation[0].resize(INPUT_NEURONS_NUMBER);
        for(auto &tmp : weightVariation[0]) tmp.resize(MIDDLE_NEURONS_NUMBER);
        weightVariation[1].resize(MIDDLE_NEURONS_NUMBER);
        for(auto &tmp : weightVariation[1]) tmp.resize(OUTPUT_NEURONS_NUMBER);
        // 初始化存放阈值的 空间
        thresholds.resize(LAYERS - 1);
        thresholds[0].resize(MIDDLE_NEURONS_NUMBER);
        thresholds[1].resize(OUTPUT_NEURONS_NUMBER);
        // 初始化存放阈值变化量的 空间
        thresholdsVariation.resize(LAYERS - 1);
        thresholdsVariation[0].resize(MIDDLE_NEURONS_NUMBER);
        thresholdsVariation[1].resize(OUTPUT_NEURONS_NUMBER);
        // 初始化存放误差项的 空间
        error.resize(LAYERS - 1);
        error[0].resize(MIDDLE_NEURONS_NUMBER);
        error[1].resize(OUTPUT_NEURONS_NUMBER);
        // 初始化存放期望的 空间
        expect.resize(OUTPUT_NEURONS_NUMBER);
    }

    /**
     * @brief 初始化权重的值[0, 1]
     * @date 2022-05-06
     */
    void initWeight(){
        for(auto &k : weight){
            for(auto &i : k){
                for(auto &j : i){
                    j = 1.0 * rand() / RAND_MAX;
                }
            }
        }
    }

    /**
     * @brief 初始化权重的变化量的值为 0
     * @date 2022-05-06
     */
    void initWeightVariation(){
        for(auto &k : weightVariation){
            for(auto &i : k){
                for(auto &j : i){
                    j = 0;
                }
            }
        }
    }

    /**
     * @brief 初始化阈值的值[0, 1]
     * @date 2022-05-06
     */
    void initThresholds(){
        for(auto &threshold : thresholds){
            for(auto &nodeThreshold : threshold){
                nodeThreshold = 1.0 * rand() / RAND_MAX;
            }
        } 
    }

    /**
     * @brief 初始化阈值的变化量的值为 0
     * @date 2022-05-06
     */
    void initThresholdsVariation(){
        for(auto &thresholdVariation : thresholdsVariation){
            for(auto &nodeThresholdVariation : thresholdVariation){
                nodeThresholdVariation = 0;
            }
        }
    }

    /**
     * @brief 初始化输入层权值
     * @date 2022-05-06
     * @param bmp 单张BMP
     */
    void initInputLayer(const BMP &bmp){
        // 取该BMP局部的像素
        unsigned char localPixelInfo[INPUT_NEURONS_NUMBER];
        int tot = 0;
        int width = bmp.bitmapInfoHeader.biWidth;
        int height = bmp.bitmapInfoHeader.biHeight;
        for(int y = UPPER_LEFT_CORNER_Y; y <= LOWER_RIGHT_CORNER_Y; y += 2){
            for(int x = UPPER_LEFT_CORNER_X; x <= LOWER_RIGHT_CORNER_X; x += 2){
                localPixelInfo[tot++] = bmp.pixelInfo[ width * (height - y - 1) + x ];
            }
        }
        // 导入输入层
        for(int i = 0; i < INPUT_NEURONS_NUMBER; i++){
            layer[0][i].in = layer[0][i].out = 1.0 * localPixelInfo[i] / UCHAR_MAX;
        }
    }

    /**
     * @brief 初始化输出层期望
     * @date 2022-05-06
     * @param owner 正在处理的图片的所有者编号
     */
    void initExcept(const int &owner){
        for(int i = 0; i < OUTPUT_NEURONS_NUMBER; i++){
            expect[i] = ( owner == i );
        }
    }

    /**
     * @brief 计算中间层
     * @date 2022-05-06
     */
    void calMiddleLayer(){
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            double nodeJInput = 0;
            for(int i = 0; i < INPUT_NEURONS_NUMBER; i++){
                nodeJInput += weight[0][i][j] * layer[0][i].out;
            }
            layer[1][j].in = nodeJInput / INPUT_NEURONS_NUMBER + thresholds[0][j];
            layer[1][j].out = f(layer[1][j].in);
            // printf("MiddleNode[%d] = {%f, %f}\n", j, layer[1][j].in, layer[1][j].out);
        }
    }

    /**
     * @brief 计算输出层
     * @date 2022-05-06
     */
    void calOutputLayer(){
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            double nodeKInput = 0;
            for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
                nodeKInput += weight[1][j][k] * layer[1][j].out;
            }
            layer[2][k].in = nodeKInput / MIDDLE_NEURONS_NUMBER + thresholds[1][k];
            layer[2][k].out = f(layer[2][k].in);
            // printf("OutputNode[%d] = {%f, %f}\n", k, layer[2][k].in, layer[2][k].out);
        }
    }

    /**
     * @brief 计算输出层误差
     * @date 2022-05-06
     */
    void calOutputError(){
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            error[1][k] = (expect[k] - layer[2][k].out) * layer[2][k].out * (1.0 - layer[2][k].out);
        }
    }

    /**
     * @brief 计算中间层误差
     * @date 2022-05-06
     */
    void calMiddleError(){
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            double tmp = 0;
            for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++) tmp += error[1][k] * weight[1][j][k];
            error[0][j] = layer[1][j].out * (1.0 - layer[1][j].out) * tmp;
        }
    }

    /**
     * @brief 迭代中间层到输出层的权值变化量
     * @date 2022-05-06
     */
    void changeMiddleToOutputWeightVariation(){
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
                weightVariation[1][j][k] = (LEARNING_EFFICIENCY / (1.0 + MIDDLE_NEURONS_NUMBER)) * (weightVariation[1][j][k] + 1.0) * error[1][k] * layer[1][j].out;
            }
        }
    }

    /**
     * @brief 迭代输入层到中间层的权值变化量
     * @date 2022-05-06
     */
    void changeInputToMiddleWeightVariation(){
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            for(int i = 0; i < INPUT_NEURONS_NUMBER; i++){
                weightVariation[0][i][j] = (LEARNING_EFFICIENCY / (1.0 + INPUT_NEURONS_NUMBER)) * (weightVariation[0][i][j] + 1.0) * error[0][j] * layer[0][i].out;
            }
        }
    }

    /**
     * @brief 迭代输出层的阈值变化量
     * @date 2022-05-06
     */
    void changeOutputThresholdsVariation(){
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            thresholdsVariation[1][k] = (LEARNING_EFFICIENCY / (1.0 + MIDDLE_NEURONS_NUMBER)) * (thresholdsVariation[1][k] + 1.0) * error[1][k];
        }
    }

    /**
     * @brief 迭代中间层的阈值变化量
     * @date 2022-05-06
     */
    void changeMiddleThresholdsVariation(){
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            thresholdsVariation[0][j] = (LEARNING_EFFICIENCY / (1.0 + INPUT_NEURONS_NUMBER)) * (thresholdsVariation[0][j] + 1.0) * error[0][j];
        }
    }

    /**
     * @brief 迭代中间层到输出层的权值
     * @date 2022-05-06
     */
    void changeMiddleToOutputWeight(){
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
                weight[1][j][k] += weightVariation[1][j][k];
            }
        }
    }

    /**
     * @brief 迭代输入层到中间层的权值
     * @date 2022-05-06
     */
    void changeInputToMiddleWeight(){
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            for(int i = 0; i < INPUT_NEURONS_NUMBER; i++){
                weight[0][i][j] += weightVariation[0][i][j];
            }
        }
    }

    /**
     * @brief 迭代输出层的阈值
     * @date 2022-05-06
     */
    void changeOutputThresholds(){
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            thresholds[1][k] += thresholdsVariation[1][k];
        }
    }

    /**
     * @brief 迭代中间层的阈值
     * @date 2022-05-06
     */
    void changeMiddleThresholds(){
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            thresholds[0][j] += thresholdsVariation[0][j];
        }
    }

    /**
     * @brief 直接计算误差
     * @date 2022-05-06
     * @return double 误差
     */
    double calError(){
        double res = 0;
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            res += (expect[k] - layer[2][k].out) * (expect[k] - layer[2][k].out);
        }
        return res / 2.0;
    }

    /**
     * @brief 训练单张BMP
     * @date 2022-05-06
     * @param bmp BMP
     * @param owner 该BMP所有者的编号
     */
    void trainingOneBmp(const BMP &bmp, const int &owner){
        // 初始化输入层的节点
        initInputLayer(bmp);
        // 初始化期望输出
        initExcept(owner);
        // 计算中间层
        calMiddleLayer();
        // 计算输出层
        calOutputLayer();
        // 计算训练前误差
        // printf("训练前的误差为: %f\n", calError());

        // 计算输出层误差项
        calOutputError();
        // 计算中间层误差项
        calMiddleError();
        // 迭代中间层到输出层的权值变化量
        changeMiddleToOutputWeightVariation();
        // 迭代输入层到中间层的权值变化量
        changeInputToMiddleWeightVariation();
        // 迭代输出层的阈值变化量
        changeOutputThresholdsVariation();
        // 迭代中间层的阈值变化量
        changeMiddleThresholdsVariation();
        // 迭代中间层到输出层的权值
        changeMiddleToOutputWeight();
        // 迭代输入层到中间层的权值
        changeInputToMiddleWeight();
        // 迭代输出层的阈值
        changeOutputThresholds();
        // 迭代中间层的阈值
        changeMiddleThresholds();
        
        // 重新计算中间层
        calMiddleLayer();
        // 重新计算输出层
        calOutputLayer();
        // 计算训练后误差
        // printf("训练后的误差为: %f\n", calError());
    }

    /**
     * @brief 载入BMP并计算误差
     * @date 2022-05-06
     * @param bmp BMP
     * @param owner 该BMP的所有者的编号
     * @return double 误差
     */
    double calOneBmpError(const BMP &bmp, const int &owner){
        // 初始化输入层
        initInputLayer(bmp);
        // 初始化期望
        initExcept(owner);
        // 计算中间层
        calMiddleLayer();
        // 计算输出层
        calOutputLayer();
        // 计算误差
        return calError();
    }

    /**
     * @brief 计算累计误差
     * @date 2022-05-06
     * @return double 累计误差
     */
    double calCumulativeError(){
        double errorSum = 0;
        for(int i = 0; i < bmpList.size(); i++){
            errorSum += calOneBmpError(bmpList[i], bmpOwnerList[i]);
        }
        return errorSum;
    }

    /**
     * @brief 保存参数到文件
     * @date 2022-05-06
     */
    void saveParameterToFile(){
        // 将各种参数写入文件
        freopen(PATAMETER_FILEPATH, "w", stdout);
        // 输出 Wij 层的权值
        for(int i = 0; i < INPUT_NEURONS_NUMBER; i++){
            for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
                printf("%f ", weight[0][i][j]);
            }
        }
        // 输出 Wjk 层的权值
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
                printf("%f ", weight[1][j][k]);
            }
        }
        // 输出 Middle 层的每个节点的阈值
        for(auto &nodeThreshold : thresholds[0]){
            printf("%f ", nodeThreshold);
        }
        // 输出 Output 层的每个节点的阈值
        for(auto &nodeThreshold : thresholds[1]){
            printf("%f ", nodeThreshold);
        }
        fclose(stdout);
        // 再次将输出重定向到控制台
        freopen("CON", "w", stdout);
        printf("参数写出完毕!!!\n");
    }

    /**
     * @brief 训练BMP集合
     * @date 2022-05-06
     */
    void trainingBMPs(){
        // 初始化各个变量空间
        initVariableSpace();
        // 初始化权重
        initWeight();
        // 初始化权重的变化量
        initWeightVariation();
        // 初始化阈值
        initThresholds();
        // 初始化阈值的变化量
        initThresholdsVariation();
        // 从文件读入参数
        readParameterFromFile();

        // 训练轮数
        int trainingRounds = MAXIMUM_TRAINING_ROUNDS;
        // 平均误差
        double averageError = DBL_MAX;
        // 训练直到 轮数结束 or 平均误差小于限定值
        while(trainingRounds-- && averageError > AVERAGE_ERROR_LIMIT){
            // 逆序训练
            // for(int i = bmpList.size() - 1; i >= 0; i--){
            //     // printf("第 %d 张图片开始训练!!!\n", i);
            //     trainingOneBmp(bmpList[i], bmpOwnerList[i]);
            // }
            // 正序训练
            for(int i = 0; i < bmpList.size(); i++){
                // printf("第 %d 张图片开始训练!!!\n", i);
                trainingOneBmp(bmpList[i], bmpOwnerList[i]);
            }
            // 计算平均误差
            averageError = calCumulativeError() / bmpList.size();
            printf("第 %d 次训练迭代的 平均误差: %f\n", MAXIMUM_TRAINING_ROUNDS - trainingRounds, averageError);
            // 每隔若干轮数保存一次参数
            if(trainingRounds % SAVE_PARAMETER_ROUNDS == 0) saveParameterToFile();
        }
        printf("总共迭代训练了 %d 次!!!\n", MAXIMUM_TRAINING_ROUNDS - trainingRounds - 1);
        // 保存参数到文件
        saveParameterToFile();
    }

    /**
     * @brief 从文件读入参数
     * @date 2022-05-06
     */
    void readParameterFromFile(){
        freopen(PATAMETER_FILEPATH, "r", stdin);
        // 输入 Wij 层的权值
        for(int i = 0; i < INPUT_NEURONS_NUMBER; i++){
            for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
                scanf("%lf", &weight[0][i][j]);
            }
        }
        // 输入 Wjk 层的权值
        for(int j = 0; j < MIDDLE_NEURONS_NUMBER; j++){
            for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
                scanf("%lf", &weight[1][j][k]);
            }
        }
        // 输入 Middle 层的每个节点的阈值
        for(auto &nodeThreshold : thresholds[0]){
            scanf("%lf", &nodeThreshold);
        }
        // 输入 Output 层的每个节点的阈值
        for(auto &nodeThreshold : thresholds[1]){
            scanf("%lf", &nodeThreshold);
        }
        fclose(stdin);
        printf("参数输入完毕!!!\n");
    }

    /**
     * @brief 处理检测结果
     * @date 2022-05-06
     * @return int 预测的所有者编号
     */
    int processResult(){
        // 各个所有者的可能性
        double possible[OUTPUT_NEURONS_NUMBER];
        double possibleSum = 0;
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            possible[k] = layer[2][k].out;
            possibleSum += possible[k];
        }
        // 寻找最大可能的所有者编号
        int maxPossibleId = 0;
        double maxPossible = 0;
        for(int k = 0; k < OUTPUT_NEURONS_NUMBER; k++){
            possible[k] /= possibleSum;
            if(possible[k] > maxPossible){
                maxPossible = possible[k];
                maxPossibleId = k;
            }
            printf("是编号为 %d 的人的概率是 %f%%\n", k + 1, possible[k] * 100.0);
        }
        return maxPossibleId + 1;
    }

    /**
     * @brief 测试一张BMP所有者的编号和实际所有者的编号是否一致
     * @date 2022-05-06
     * @param bmp BMP
     * @param owner 实际所有者
     * @return true 相同
     * @return false 不同
     */
    bool testOneBmp(const BMP &bmp, const int &owner){
        // 初始化输入层的节点
        initInputLayer(bmp);
        // 初始化期望输出
        initExcept(owner);
        // 计算中间层
        calMiddleLayer();
        // 计算输出层
        calOutputLayer();
        // 处理输出层结果
        int maxPossibleId = processResult();
        printf("最大可能的人员编号: %d, 期望的人员编号是: %d, 误差为: %f\n", maxPossibleId, owner + 1, calError());
        return maxPossibleId == owner + 1;
    }

    /**
     * @brief 测试BMP集合
     * @date 2022-05-06
     */
    void testBMPs(){
        // 初始化各个变量空间
        initVariableSpace();
        // 读入参数
        readParameterFromFile();
        // 判断成功的BMP个数
        int sameCount = 0;
        for(int i = 0; i < bmpList.size(); i++){
            printf("处理文件路径为 %s 的图片...\n", bmpFilePathList[i]);
            sameCount += testOneBmp(bmpList[i], bmpOwnerList[i]);
            printf("\n");
        }
        printf("全部测试完毕!!!\n");
        printf("判断正确数量为: %d / %d\n", sameCount, bmpList.size());
    }

};

/**
 * @brief 初始化训练集的文件路径集合
 * @date 2022-05-06
 * @param filePathList 
 */
void initTrainingFilePathList(vector<const char *> &filePathList){
    string pathPrefix = "YALE\\\\subject";
    string pathSuffix = ".bmp";
    for(int i = 1; i <= 15; i++){
        string tmp = pathPrefix;
        if(i < 10) tmp += '0', tmp += ( i + '0');
        else tmp += (i / 10 + '0'), tmp += (i % 10 + '0');
        tmp += '_';
        for(int j = 1; j <= 5; j++){
            string tmp2 = tmp;
            tmp2 += (j + '0');
            tmp2 += pathSuffix;
            char *s = (char *)malloc(sizeof(char) * 20);
            strcpy(s, tmp2.c_str());
            filePathList.push_back(s);
        }
    }
}

/**
 * @brief 初始化测试集的文件路径集合
 * @date 2022-05-06
 * @param filePathList 
 */
void initTestFilePathList(vector<const char *> &filePathList){
    string pathPrefix = "YALE\\\\subject";
    string pathSuffix = ".bmp";
    for(int i = 1; i <= 15; i++){
        string tmp = pathPrefix;
        if(i < 10) tmp += '0', tmp += ( i + '0');
        else tmp += (i / 10 + '0'), tmp += (i % 10 + '0');
        tmp += '_';
        for(int j = 6; j <= 11; j++){
            string tmp2 = tmp;
            if(j < 10) tmp2 += (j + '0');
            else tmp2 += (j / 10 + '0'), tmp2 += (j % 10 + '0');
            tmp2 += pathSuffix;
            char *s = (char *)malloc(sizeof(char) * 20);
            strcpy(s, tmp2.c_str());
            filePathList.push_back(s);
        }
    }
}

/**
 * @brief main
 * @date 2022-05-06
 * @return int 0 
 */
int main(){
    srand(time(0));
    vector<const char *> filePathList;
    // initTrainingFilePathList(filePathList);
    // BPNeuralNetwork bp(filePathList);
    // bp.trainingBMPs();

    // initTrainingFilePathList(filePathList);
    initTestFilePathList(filePathList);
    BPNeuralNetwork bp(filePathList);
    bp.testBMPs();

    return 0;
}