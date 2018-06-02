#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;
using namespace cv;

//looks like there's a problem with the 7 and 8 dataset?

string IntToString (int a)
{
    ostringstream temp;
    temp<<a;
    return temp.str();
}

cv::Mat positionNormalisation(cv::Mat src, int thresh1, int thresh2)
{
	cv::Mat thr, transMat, transLat;
	int count = 0, sumx = 0, sumy = 0, imMag, tranx = 0, trany = 0;

	cv::threshold(src, thr, thresh1, thresh2, cv::THRESH_BINARY);
	cv::bitwise_not(thr, thr);
	for(int i=0; i<src.cols; i++)
	{
		for(int j=0; j<src.rows; j++)
		{
			imMag = (int)thr.at<uchar>(j, i);
			if(imMag == 255)
			{
				sumx+=i;
				sumy+=j;
				count++;
			}
		}
	}

	if(count>0)
	{
		sumx=sumx/count;
		sumy=sumy/count;
		cv::Point p(sumx, sumy);

		tranx = (src.cols/2)-sumx;
		trany = (src.rows/2)-sumy;

		transMat = (cv::Mat_<double>(2,3) << 1, 0, tranx, 0, 1, trany);
		cv::warpAffine(src, transLat, transMat, cv::Size(src.cols, src.rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	}
	else
	{
		return src;
	}	
//	cout << sumx << " " << sumy << endl;

	
	return transLat;
}

cv::Mat imageSquash(cv::Mat src, float scale)
{
	cv::Mat squashMat;
//	cout << "test" << endl;
	if(scale>1)
	{
		float sides = (src.cols-src.cols/scale)/2;
	//	cout << sides << endl;
		cv::resize(src, squashMat, cv::Size(src.cols/scale, src.rows));
		cv::copyMakeBorder(squashMat, squashMat, 0, 0, (int)sides, (int)sides, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));	
	}
	else if(scale<1)
	{
		float sides = (src.cols/scale-src.cols)/2;
	//	cout << sides << endl;
		cv::resize(src, squashMat, cv::Size(src.cols/scale, src.rows));
		cv::Rect myROI(sides, 0, squashMat.cols-((int)sides*2), squashMat.rows);
		squashMat = squashMat(myROI);
	//	cout << sides << endl;
	//	cout << squashMat.cols << endl;
	//	cout << squashMat.rows << endl;
	}
	else
	{
		return src;
	}
	return squashMat;
}

cv::Mat imageRotate(cv::Mat src, float angle)
{
	cv::Mat rotateMat, rMat;
	cv::Point2f pt(src.cols/2, src.rows/2);
	rMat = cv::getRotationMatrix2D(pt, angle, 1.0);
	cv::warpAffine(src, rotateMat, rMat, Size(src.cols, src.rows), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	return rotateMat;
}

int main(int args, char** argv)
{
	std::string inputDirec = "Dataset_Digits/";
	std::string outputDirec = "ArtificialDigits/";
	std::string extension = ".jpg";
	float stretch [4] = {1.75, 1.4, 0.77, 0.7};
	float angles [4] = {15, 30, 345, 330};
	int count;
	cv::Mat image, norm, stretchMat;
	ofstream labels;

	labels.open("ArtificialDigits/labels.txt");
	labels << "#NAME, DIGIT CLASS\n";
	for(int i=0; i<10; i++)
	{
		count = 0;
		std::string currFolder = IntToString(i);
		for(int j=0; j<46; j++)
		{
			std::string temp1 = inputDirec+currFolder+"/"+currFolder+"_";
			if(j<10)
			{
				temp1.append("0");
			}
		
			temp1.append(IntToString(j));
			temp1.append(extension);
			image = imread(temp1, CV_LOAD_IMAGE_GRAYSCALE);
			norm = positionNormalisation(image, 150, 255);

			cv::imwrite(outputDirec+currFolder+"/"+currFolder+"_"+IntToString(count)+extension, norm);
			labels << currFolder+"_"+IntToString(count)+extension+" "+currFolder+"\n";
			count++;
	//		cout << "a " << count << endl;
			for(int k=0; k<4; k++)
			{
				image = imageRotate(norm, angles[k]);
				cv::imwrite(outputDirec+currFolder+"/"+currFolder+"_"+IntToString(count)+extension, image);
				labels << currFolder+"_"+IntToString(count)+extension+" "+currFolder+"\n";
				count++;
	//			cout << "b " << count << endl;
			}
			for(int k=0; k<4; k++)
			{
				stretchMat = imageSquash(norm, stretch[k]);
				cv::imwrite(outputDirec+currFolder+"/"+currFolder+"_"+IntToString(count)+extension, stretchMat);
				labels << currFolder+"_"+IntToString(count)+extension+" "+currFolder+"\n";
				count++;
	//			cout << "c " << count << endl;
				for(int l=0; l<4; l++)
				{
					image = imageRotate(stretchMat, angles[l]);
					cv::imwrite(outputDirec+currFolder+"/"+currFolder+"_"+IntToString(count)+extension, image);
					labels << currFolder+"_"+IntToString(count)+extension+" "+currFolder+"\n";
					count++;
	//				cout << "d " << count << endl;
				}
			}
		}
	}
	labels.close();
//	return 0;
}