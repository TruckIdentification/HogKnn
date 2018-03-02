#include "opencv/cv.h" 
#include <opencv2\opencv.hpp>  
#include <stdio.h>  
#include <iostream> 
#include <opencv2/highgui/highgui.hpp>  
#include <opencv/ml.h>   
#include <iostream>    
#include <fstream>    
#include <string>    
#include <vector>  
using namespace cv;    
using namespace std;
Mat MoveDetect(Mat background, Mat frame);
Point originalPoint(900, 350); //矩形框起点  
Point processPoint(1400, 800);
int ImgWidht=28; 
int ImgHeight=28; 
CvKNearest knn;
int FindK=3;  //KNN算法中找5个最近点
int choose;
void show();
void train(string s);
int test(Mat s);

int main(int argc, char** argv)      
{
	int flag = 0;
	train("train_car3.txt");
	VideoCapture capture("1111.mp4");
	int m, n, j;
	m = n = j = 0;
	Mat background;//存储背景图像  
	Mat result;//存储结果图像  
	while(1)
	{
		
		//show();
		 
		//train("train_car3.txt");

		Mat frame;//定义一个Mat变量，用于存储每一帧的图像
		capture >> frame;  //读取当前帧
		rectangle(frame, originalPoint, processPoint, Scalar(255, 0, 0), 3, 4, 0);
		Mat imageROI = frame(Rect(450, 150, 500, 450));  //(900, 350, 500, 450)
		imshow("读取视频", frame);  //显示当前帧

		//test("test_car3.txt");
		

		int a = test(imageROI);

		
		if (a == 0){
			n++;
			if (n >= 10 && m < 10){
				flag = 1;
				j = 1;
				cout << "OKOKOKOKOKOKOKOKOK" << endl;
				cout << "n===" << n << "     " << "M====" << m << endl;
				imshow("roi", imageROI);
			}
		}
		else{
			m++;
			if (m > 10){
				m = 0;
				n = 0;
				
			}
			if (flag == 1){

				background = imageROI;
				flag++;
			}
			if (j > 0){
				//result = MoveDetect(background, imageROI);//调用MoveDetect()进行运动物体检测，返回值存入result 
				//imshow("result", result);
			}
			
		}

		cout << a << endl;
		/*if (a==0){

			imshow("roi", imageROI);
		}*/

		waitKey(30);  //延时30ms


		 
	}       
    return 0;    
} 
/*void show()
{
	cout<<"****************************"<<endl;
	cout<<"******1:人与背景分类实验******"<<endl;
	cout<<"******2:车与背景分类实验******"<<endl;
	cout<<"****************************"<<endl;
	
	cout<<"请输入1 或 2:";
	cin>>choose;
	cout<<"****************************"<<endl;
}*/
void train(string filename)
{
	vector<string> src_img_path;    //图片名字
    vector<int> img_class;          //图片样本类别  0代表正样本  1代表负样本
	ifstream svm_train_img(filename); 
	string templine;     
    int trainline = 0;       
    unsigned long n;    
    while(svm_train_img)      //读取文件名进入vector中
    {    
        if(getline(svm_train_img,templine))    
        {    
            trainline++;    
            if(trainline<201)     //前面100张是正样本，分类为0
            {    
                img_class.push_back(0);  
                src_img_path.push_back(templine);//图像路径   
            }    
            else             //之后的是负样本，分类为1
            {    
                img_class.push_back(1);  
                src_img_path.push_back(templine);//图像路径   
            }    
        }    
    }    
    svm_train_img.close();    //关闭文件    

    Mat train_hog, train_class;        
    train_class=Mat::zeros(trainline,1,CV_32FC1 );      //样本类别矩阵
	cout<<"提取直方图特征!"<<endl;
    for(string::size_type i=0;i!=src_img_path.size();i++)    
    {    
		cout<<src_img_path[i].c_str()<<endl; 
        Mat src_img=imread(src_img_path[i].c_str(), 1);      
//		cout<<src_img<<endl;
        resize(src_img,src_img,cv::Size(ImgWidht,ImgHeight),0,0,INTER_CUBIC);   //图片大小调整
		cout<<"读取图片并重新定义大小完成!"<<endl;
        HOGDescriptor *hog=new HOGDescriptor(cvSize(ImgWidht,ImgHeight),cvSize(14,14),cvSize(7,7),cvSize(7,7), 9);  //提取直方图特征
		//第一个参数是窗口大小，第二个是块大小，第三个是块步长，第四个是包元大小
		//64*128 16*16 8*8
		//（（64-16）/8+1）*（（126-16）/8+1）=105
		//(16/8)*(16/8)=4个鲍元
		//9维
		//共9*4*105个特征
        vector<float> hog_result;     
        hog->compute(src_img,hog_result,Size(1,1),Size(0,0)); //直方图计算
		cout<<"梯度颜色直方图计算完成!"<<endl;
        if (i==0)    //第一张图片的时候分配直方图Mat,每一行代表一个样本
        {  
             train_hog = Mat::zeros(trainline,hog_result.size(),CV_32FC1 );  
			 cout<<"初始化创建存储矩阵!"<<endl;
        }     
        n=0;    
        for(vector<float>::iterator iter=hog_result.begin();iter!=hog_result.end();iter++)     //利用迭代器进行Mat数组赋值
        {    
            train_hog.at<float>(i,n)=*iter;    
            n++;    
        }      
		cout<<"矩阵元素提取完成!"<<endl;
        train_class.at<float>(i,0)=img_class[i];     
    }             
	cout<<"**************************"<<endl;
	cout<<"开始训练KNN!"<<endl;

    knn.train(train_hog,train_class,Mat(),false,FindK);  

	cout<<"训练结束!"<<endl;
	cout<<"**************************"<<endl;
}
int test(Mat ROI)
{
	//cout<<"**************************"<<endl; 
	//cout<<"开始测试!"<<endl;
    /*vector<string> img_test_path;  
    ifstream svm_test_img(filename);   
	string templine; 
    while(svm_test_img)    
    {    
        if(getline(svm_test_img,templine))    
        {    
           img_test_path.push_back(templine);            
		}    
    }    
    svm_test_img.close();    
	*/

    //int rightnum=0;
     
    char line[512];    
    ofstream predict("predict_data.txt");    
    /*for(string::size_type j=0;j!=img_test_path.size();j++ )    
    {   */ 
		//cout<<img_test_path[j].c_str()<<endl; 
        //Mat src_image=imread(img_test_path[j].c_str(),1);//读入图像     
		Mat src_image = ROI;
        resize(src_image, src_image, cv::Size(ImgWidht,ImgHeight), 0, 0, INTER_CUBIC);//要搞成同样的大小才可以检测到 

        HOGDescriptor *hog=new HOGDescriptor(cvSize(ImgWidht,ImgHeight),cvSize(14,14),cvSize(7,7),cvSize(7,7),9); 

        vector<float> hog_result;//结果数组    

        hog->compute(src_image,hog_result,Size(1,1),Size(0,0));      
        Mat test_hog=Mat::zeros(1,hog_result.size(),CV_32FC1); 
		int n=0;
        for(vector<float>::iterator iter=hog_result.begin();iter!=hog_result.end();iter++)    
        {    
            test_hog.at<float>(0,n) = *iter;    
            n++;    
        }    
        int result = knn.find_nearest(test_hog,FindK);     //利用支持向量机测试
	    //cout<<"The perdict class: "<<result; 
		return int(result);
	   /* if(j<27)
	    {
			cout<<"\tThe right class: 0   ";
		    if(result==0)
		    {
			    cout<<"\tright"<<endl;
			    rightnum++;
		    }
		    else
			    cout<<"\twrong"<<endl;
	    }
	    else
	    {
			cout<<"\tThe right class: 1    ";
		    if(result==1)
		    {
			    cout<<"\tright"<<endl;
			    rightnum++;
		    }
		    else
			    cout<<"\twrong"<<endl;
	    }
        predict<<line;    
       }
	/*cout<<"样本测试完成!"<<endl;
	cout<<"当前选择特征与分类器:HOG+KNN"<<endl;
	cout<<"总体预测准确率: "<<rightnum*100.0/127<<"%"<<endl<<endl<<endl;
    predict.close(); */
}

Mat MoveDetect(Mat background, Mat frame)
{
	Mat result = frame.clone();
	//1.将background和frame转为灰度图  
	Mat gray1, gray2;
	cvtColor(background, gray1, CV_BGR2GRAY);
	cvtColor(frame, gray2, CV_BGR2GRAY);
	//2.将background和frame做差  
	Mat diff;
	absdiff(gray1, gray2, diff);
	imshow("diff", diff);
	//3.对差值图diff_thresh进行阈值化处理  
	Mat diff_thresh;
	threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
	imshow("diff_thresh", diff_thresh);
	//4.腐蚀  
	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(diff_thresh, diff_thresh, kernel_erode);
	imshow("erode", diff_thresh);
	//5.膨胀  
	dilate(diff_thresh, diff_thresh, kernel_dilate);
	imshow("dilate", diff_thresh);
	//6.查找轮廓并绘制轮廓  
	vector<vector<Point>> contours;
	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓  
	//7.查找正外接矩形  
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形  
	}
	return result;//返回result  
}