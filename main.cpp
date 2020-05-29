#include <bits/stdc++.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

typedef pair<int,int> ii;
typedef pair<float, float> ff;

///////////////////////////////////////////////////////////////////

//Fixed parameters used
int patchX = 7, patchY = 7, rmax = max(patchX, patchY), kmax = 10;
float rho = 0.5, lambda = 50.0, sigma;

///////////////////////////////////////////////////////////////////

//Auxiliar function to see the type used for pixels and number of channels
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

///////////////////////////////////////////////////////////////////

//Auxiliar function to convert an image from 8UC3 to 32FC3 (in the case of the original image) 
//or from 16SC1 to 32FC1 (in the case of textures)
Mat char_to_float(Mat &I){
	Mat I_transf;

	if(type2str(I.type()) == "8UC3"){
		I_transf = Mat::zeros(I.rows, I.cols, CV_32FC3);

		for(int i=0;i<I.rows;i++){
			for(int j=0;j<I.cols;j++){
				I_transf.at<Vec3f>(i,j).val[0] = I.at<Vec3b>(i,j).val[0]/255.0;
				I_transf.at<Vec3f>(i,j).val[1] = I.at<Vec3b>(i,j).val[1]/255.0;
				I_transf.at<Vec3f>(i,j).val[2] = I.at<Vec3b>(i,j).val[2]/255.0;
			}
		}
	}
	else if(type2str(I.type()) == "16SC1"){
		I_transf = Mat::zeros(I.rows, I.cols, CV_32FC1);

		for(int i=0;i<I.rows;i++){
			for(int j=0;j<I.cols;j++){
				I_transf.at<float>(i,j) = (1.0*I.at<int>(i,j) - 1.0*numeric_limits<int>::min())/(1.0*numeric_limits<int>::max() - 1.0*numeric_limits<int>::min());
			}
		}
	}

	return I_transf;
}

///////////////////////////////////////////////////////////////////

//Function to get the expanded occlusion using the occlusion as input
Mat get_expanded_occ(Mat &occ_input){
	Mat expanded_occ = Mat::zeros(occ_input.rows, occ_input.cols, CV_8UC1);

	for(int i=0;i<occ_input.rows;i++){
		for(int j=0;j<occ_input.cols;j++) expanded_occ.at<uchar>(i,j) = occ_input.at<uchar>(i,j);
	}

	for(int i=0;i<occ_input.rows - patchX;i++){
		for(int j=0;j<occ_input.cols - patchY;j++){
			bool inside = false, outside = false;
			Mat window_image(occ_input, Rect(j, i, patchX, patchY));

			for(int k=0;k<window_image.rows;k++){
				for(int m=0;m<window_image.cols;m++){
					if(window_image.at<uchar>(k,m) == 0) outside = true;
					else if(window_image.at<uchar>(k,m) == 255) inside = true; 
				}
			}

			if(inside && outside) expanded_occ.at<uchar>(i + patchX/2,j + patchY/2) = 255; 
		}
	}

	return expanded_occ;
}

///////////////////////////////////////////////////////////////////

//Function to upsample an image using gaussian pyramid
Mat upsample(Mat &I, float factor){
	Mat tmp = I, dst = I;

	pyrUp( tmp, dst, Size( tmp.cols*factor, tmp.rows*factor ) );

	return dst;
}

///////////////////////////////////////////////////////////////////

//Function to subsample an image using gaussian pyramid
Mat subsample(Mat &I, float factor){
	Mat tmp = I, dst = I;

	pyrDown( tmp, dst, Size( tmp.cols/factor, tmp.rows/factor ));

	return dst;
}

///////////////////////////////////////////////////////////////////

//Function to get the gaussian pyramid. Each element of the resulting vector is a level of the pyramid
vector<Mat> get_pyramid(Mat &I, int levels){
	Mat tmp = I, dst = I;
	vector<Mat> pyramid;

	int level = 0;
	pyramid.push_back(dst);

	level++;

	while(level < levels){
		dst = subsample(tmp, 2);
		pyramid.push_back(dst);
		level++;
		tmp = dst;
	}

	return pyramid;
}

///////////////////////////////////////////////////////////////////

//Function to pass erosion operator through the image
Mat get_eroded(Mat &occ_input){
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

	Mat copy = occ_input, dst;
	erode(copy, dst, element);

	return dst;
}

///////////////////////////////////////////////////////////////////

//Auxiliar function to see if the occlusion is completely erased.
bool occ_erased(Mat &occ_input){
	for(int i=0;i<occ_input.rows;i++){
		for(int j=0;j<occ_input.cols;j++){
			if(occ_input.at<uchar>(i,j) == 255) return false;
		}
	}

	return true;
}

///////////////////////////////////////////////////////////////////

//Function to get ideal number of levels for the gaussian pyramid
int get_levels(Mat &occ_input){
	Mat occ_copy = occ_input, occ_gray;

	cvtColor(occ_copy, occ_gray, COLOR_BGR2GRAY);

	int N = max(patchX, patchY), No = 0;

	Mat tmp = occ_gray, dst;

	while(!occ_erased(tmp)){
		No++;
		dst = get_eroded(tmp);
		tmp = dst;
	}

	No *= 2;

	return (int)floor((float)(log(((float) No)/((float)N)))/((float)log(2)));
}

///////////////////////////////////////////////////////////////////

//Function to calculate distance between neighbors patches given the pixel position of the center of the patch
//and given the shift map value for that pixel
float dist_patch(Mat &I, int x, int y, int shiftX, int shiftY){
	float d = 0.0;

	for(int i=-3;i<=3;i++){
		for(int j=-3;j<=3;j++){
			if(x+i < 0 || x+i >= I.rows || y+j < 0 || y+j >= I.cols || x+shiftX+i < 0 || x+shiftX+i >= I.rows || y+shiftY+j < 0 || y+shiftY+j >= I.cols) continue;

			d += (I.at<Vec3f>(x+i,y+j).val[0] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[0])*(I.at<Vec3f>(x+i,y+j).val[0] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[0]);
			d += (I.at<Vec3f>(x+i,y+j).val[1] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[1])*(I.at<Vec3f>(x+i,y+j).val[1] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[1]);
			d += (I.at<Vec3f>(x+i,y+j).val[2] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[2])*(I.at<Vec3f>(x+i,y+j).val[2] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[2]);
		}
	}

	return d;
}

///////////////////////////////////////////////////////////////////

//Function to get the random uniform variable in [-1,1]², which is useful for PatchMatch algorithm
ff RandUnif(float a, float b){
	random_device rd;
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(a, b);

    ff theta;
    theta.first = dis(gen);
    theta.second = dis(gen);

    return theta;
}

///////////////////////////////////////////////////////////////////

//PatchMatch algorithm: the shift map values are updated after the running of the algorithm
void patch_match(Mat &I, Mat &occ_input, vector< vector<ii> > &shift_map){
	Mat expanded_occ = get_expanded_occ(occ_input);

	for(int k=1;k<=kmax;k++){
		for(int i=0;i<expanded_occ.rows;i++){
			for(int j=0;j<expanded_occ.cols;j++){
				if(expanded_occ.at<uchar>(i,j) == 255){
					int px = i, py = j, ax, ay, bx, by, rx, ry;

					if(k%2 == 0){
						ax = i-1; ay = j; bx = i; by = j-1; rx = i; ry = j;

						if(ax < 0) ax++;
						if(by < 0) by++;

						float distp = dist_patch(I, i, j, shift_map[i][j].first, shift_map[i][j].second);
						float dista = dist_patch(I, i, j, shift_map[ax][ay].first, shift_map[ax][ay].second);
						float distb = dist_patch(I, i, j, shift_map[bx][by].first, shift_map[bx][by].second);

						float dMin = distp;

						if(dista < dMin){
							dMin = dista;
							rx = ax;
							ry = ay;
						}

						if(distb < dMin){
							dMin = distb;
							rx = bx;
							ry = by;
						}

						if(expanded_occ.at<uchar>(i + shift_map[rx][ry].first,j + shift_map[rx][ry].second) == 0)
							shift_map[i][j] = shift_map[rx][ry];
					}
					else{
						px = expanded_occ.rows - i - 1;
						py = expanded_occ.cols - j - 1;
						ax = px+1; ay = py; bx = px; by = py+1; rx = px; ry = py;

						if(ax >= expanded_occ.rows) ax--;
						if(by >= expanded_occ.cols) by--;

						float distp = dist_patch(I, px, py, shift_map[px][py].first, shift_map[px][py].second);
						float dista = dist_patch(I, px, py, shift_map[ax][ay].first, shift_map[ax][ay].second);
						float distb = dist_patch(I, px, py, shift_map[bx][by].first, shift_map[bx][by].second);

						float dMin = distp;

						if(dista < dMin){
							dMin = dista;
							rx = ax;
							ry = ay;
						}

						if(distb < dMin){
							dMin = distb;
							rx = bx;
							ry = by;
						}

						if(expanded_occ.at<uchar>(i + shift_map[rx][ry].first,j + shift_map[rx][ry].second) == 0)
							shift_map[i][j] = shift_map[rx][ry];
					}

					int zmax = (int)ceil((float)(-(log((float)(rmax)))/(log((float)(rho)))));

					for(int z=1;z<=zmax;z++){
						ff theta = RandUnif(-1.0, 1.0);
						int randomShiftX = (int)floor(((float)rmax)*((float)pow(rho, (float)z))*(theta.first));
						int randomShiftY = (int)floor(((float)rmax)*((float)pow(rho, (float)z))*(theta.second));

						
						int qx = px + shift_map[px][py].first + randomShiftX;
						int qy = py + shift_map[px][py].second + randomShiftY;

						if(!(qx < 0 || qx >= I.rows || qy < 0 || qy >= I.cols || px + shift_map[qx][qy].first < 0 
							|| px + shift_map[qx][qy].first >= I.rows || py + shift_map[qx][qy].second < 0 
							|| py + shift_map[qx][qy].second >= I.cols || px + shift_map[px][py].first < 0 
							|| px + shift_map[px][py].first >= I.rows || py + shift_map[px][py].second < 0 
							|| py + shift_map[px][py].second >= I.cols) 
							&& expanded_occ.at<uchar>(px + shift_map[qx][qy].first, py + shift_map[qx][qy].second) == 0
							&& dist_patch(I, px, py, shift_map[qx][qy].first, shift_map[qx][qy].second) < dist_patch(I, px, py, shift_map[px][py].first, shift_map[px][py].second))
							shift_map[px][py] = shift_map[qx][qy];
					}
				}
			}
		}
	}
}

///////////////////////////////////////////////////////////////////

//Function to get the sigma value used during the reconstruction step
float get_sigma(Mat &I, vector< vector<ii> > &shift_map){
	vector<float> dists;

	int cont = 0;
	for(int i=0;i<I.rows;i++){
		for(int j=0;j<I.cols;j++){
			float dist = dist_patch(I, i, j, shift_map[i][j].first, shift_map[i][j].second);
			dists.push_back(dist);
		}
	}

	sort(dists.begin(), dists.end());

	int N = (int)dists.size();
	int ind = (int)ceil(0.75*N);

	return dists[ind-1];
}

///////////////////////////////////////////////////////////////////

//Function to get the weight used during the reconstruction step
float get_weight(Mat &I, int x, int y, int neighborX, int neighborY, vector< vector<ii> > &shift_map){
	return (float)(exp(-((dist_patch(I, neighborX, neighborY, shift_map[neighborX][neighborY].first, shift_map[neighborX][neighborY].second))/(2*sigma*sigma))));
}

///////////////////////////////////////////////////////////////////

//Function that executes the reconstruction step
void pixel_reconstruction(Mat &I, int px, int py, vector< vector<ii> > &shift_map){
	Vec3f num;
	float den = 0.0;

	num.val[0] = 0.0;
	num.val[1] = 0.0;
	num.val[2] = 0.0;

	for(int i=-3;i<=3;i++){
		for(int j=-3;j<=3;j++){
			if(px+i < 0 || px+i >= I.rows || py+j < 0 || py+j >= I.cols) continue;

			int qx = px+i, qy = py+j;

			if(px + shift_map[qx][qy].first < 0 || px + shift_map[qx][qy].first >= I.rows ||
				py + shift_map[qx][qy].second < 0 || py + shift_map[qx][qy].second >= I.cols) continue;

			float spq = get_weight(I, px, py, qx, qy, shift_map);
			den += spq;
			num.val[0] += spq*I.at<Vec3f>(px + shift_map[qx][qy].first, py + shift_map[qx][qy].second).val[0];
			num.val[1] += spq*I.at<Vec3f>(px + shift_map[qx][qy].first, py + shift_map[qx][qy].second).val[1];
			num.val[2] += spq*I.at<Vec3f>(px + shift_map[qx][qy].first, py + shift_map[qx][qy].second).val[2];
		}
	}

	num.val[0] /= den;
	num.val[1] /= den;
	num.val[2] /= den;

	I.at<Vec3f>(px, py).val[0] = num.val[0];
	I.at<Vec3f>(px, py).val[1] = num.val[1];
	I.at<Vec3f>(px, py).val[2] = num.val[2];
}

///////////////////////////////////////////////////////////////////

//Function to do the final reconstruction when we are in the base of the gaussian pyramid
void final_reconstruction(Mat &I, int px, int py, vector< vector<ii> > &shift_map){
	float dMin = -1.0;
	ii argMin;

	if(!(px + shift_map[px][py].first < 0 || px + shift_map[px][py].first >= I.rows 
		|| py + shift_map[px][py].second < 0 || py + shift_map[px][py].second >= I.cols)){
		I.at<Vec3f>(px, py).val[0] = I.at<Vec3f>(px + shift_map[px][py].first, py + shift_map[px][py].second).val[0];
		I.at<Vec3f>(px, py).val[1] = I.at<Vec3f>(px + shift_map[px][py].first, py + shift_map[px][py].second).val[1];
		I.at<Vec3f>(px, py).val[2] = I.at<Vec3f>(px + shift_map[px][py].first, py + shift_map[px][py].second).val[2];
	}

	for(int i=-3;i<=3;i++){
		for(int j=-3;j<=3;j++){
			if(px+i < 0 || px+i >= I.rows || py+j < 0 || py+j >= I.cols 
				|| px+i+shift_map[px+i][py+j].first < 0 || px+i+shift_map[px+i][py+j].first >= I.rows 
				|| py+j+shift_map[px+i][py+j].second < 0 || py+j+shift_map[px+i][py+j].second >= I.cols
				//|| px+shift_map[px+i][py+j].first < 0 || px+shift_map[px+i][py+j].first >= I.rows 
				//|| py+shift_map[px+i][py+j].second < 0 || py+shift_map[px+i][py+j].second >= I.cols
				) continue;
			else if(dMin == -1.0 || dist_patch(I, px+i, py+j, shift_map[px+i][py+j].first, shift_map[px+i][py+j].second) < dMin){
				dMin = dist_patch(I, px+i, py+j, shift_map[px+i][py+j].first, shift_map[px+i][py+j].second);
				argMin = {px+i, py+j};
			}
		}
	}
	
	if(!(px + shift_map[argMin.first][argMin.second].first < 0 
		|| px + shift_map[argMin.first][argMin.second].first >= I.rows 
		|| py + shift_map[argMin.first][argMin.second].second < 0 
		|| py + shift_map[argMin.first][argMin.second].second >= I.cols)){
		I.at<Vec3f>(px, py).val[0] = I.at<Vec3f>(px + shift_map[argMin.first][argMin.second].first, py + shift_map[argMin.first][argMin.second].second).val[0];
		I.at<Vec3f>(px, py).val[1] = I.at<Vec3f>(px + shift_map[argMin.first][argMin.second].first, py + shift_map[argMin.first][argMin.second].second).val[1];
		I.at<Vec3f>(px, py).val[2] = I.at<Vec3f>(px + shift_map[argMin.first][argMin.second].first, py + shift_map[argMin.first][argMin.second].second).val[2];
	}
}

///////////////////////////////////////////////////////////////////

//Function to get a random integer in the interval [1, up_limit]
int RandUnifInt(int up_limit){
	random_device rd;
    mt19937 gen(rd()); 
    uniform_int_distribution<int> dis(1, up_limit);
    return dis(gen);
}

///////////////////////////////////////////////////////////////////

//Function to do random initialization of the shift map
void initialize_shift_map(Mat &occ_input, vector< vector<ii> > &shift_map){
	int rows = (int)shift_map.size();
	int cols = (int)shift_map[0].size();

	for(int i=0;i<(int)shift_map.size();i++){
		for(int j=0;j<(int)shift_map[i].size();j++){
			shift_map[i][j].first = RandUnifInt(rows-1) - i;
			shift_map[i][j].second = RandUnifInt(cols-1) - j;

			while(occ_input.at<uchar>(i + shift_map[i][j].first, j + shift_map[i][j].second) == 255){
				shift_map[i][j].first = RandUnifInt(rows-1) - i;
				shift_map[i][j].second = RandUnifInt(cols-1) - j;
			}
		}
	}
}

///////////////////////////////////////////////////////////////////

//Function to upsample the shift map through close levels in the gaussian pyramid
vector< vector<ii> > get_upsampled_shift_map(vector< vector<ii> > &shift_map){
	int rows = (int)shift_map.size();
	int cols = (int)shift_map[0].size();
	vector< vector<ii> > upsampled_shift_map(2*rows, vector<ii>(2*cols));

	for(int i=0;i<rows;i++){
		for(int j=0;j<cols;j++){
			upsampled_shift_map[2*i][2*j] = shift_map[i][j];
			upsampled_shift_map[2*i+1][2*j] = shift_map[i][j];
			upsampled_shift_map[2*i][2*j+1] = shift_map[i][j];
			upsampled_shift_map[2*i+1][2*j+1] = shift_map[i][j];
		}
	}

	return upsampled_shift_map;
}

///////////////////////////////////////////////////////////////////

//Function to test if a pixel is on the border of the occlusion
bool is_border(Mat &cur_occ, int row, int col){
	if(cur_occ.at<uchar>(row,col) == 0) return false;

	for(int i=-1;i<=1;i++){
		for(int j=-1;j<=1;j++){
			if((i == 0 && j == 0) || (row+i < 0) || (row+i >= (int)cur_occ.rows) || (col+j < 0) || (col+j >= (int)cur_occ.cols)) continue;
			if(cur_occ.at<uchar>(row+i, col+j) == 0) return true;
		}
	}

	return false;
}

///////////////////////////////////////////////////////////////////

//Function to calculate distance between neighbors patches during the onion-peel initialization
float dist_patch_init(Mat &I, Mat &cur_occ, int x, int y, int shiftX, int shiftY){
	float d = 0.0;

	for(int i=-3;i<=3;i++){
		for(int j=-3;j<=3;j++){
			if(x+i < 0 || x+i >= I.rows || y+j < 0 || y+j >= I.cols && x+shiftX+i < 0 || x+shiftX+i >= I.rows || y+shiftY+j < 0 || y+shiftY+j >= I.cols) continue;

			if(cur_occ.at<uchar>(x+i,y+j) == 0){
				d += (I.at<Vec3f>(x+i,y+j).val[0] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[0])*(I.at<Vec3f>(x+i,y+j).val[0] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[0]);
				d += (I.at<Vec3f>(x+i,y+j).val[1] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[1])*(I.at<Vec3f>(x+i,y+j).val[1] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[1]);
				d += (I.at<Vec3f>(x+i,y+j).val[2] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[2])*(I.at<Vec3f>(x+i,y+j).val[2] - I.at<Vec3f>(x+shiftX+i,y+shiftY+j).val[2]);
			}
		}
	}

	return d;
}

///////////////////////////////////////////////////////////////////

//Function to do pixel reconstruction during the onion-peel initialization
void pixel_reconstruction_init(Mat &I, Mat &cur_occ, int px, int py, vector< vector<ii> > &shift_map){
	Vec3f num;
	float den = 0.0;

	num.val[0] = 0.0;
	num.val[1] = 0.0;
	num.val[2] = 0.0;

	for(int i=-3;i<=3;i++){
		for(int j=-3;j<=3;j++){
			if(px+i < 0 || px+i >= I.rows || py+j < 0 || py+j >= I.cols) continue;

			int qx = px+i, qy = py+j;

			if(px + shift_map[qx][qy].first < 0 || px + shift_map[qx][qy].first >= I.rows ||
				py + shift_map[qx][qy].second < 0 || py + shift_map[qx][qy].second >= I.cols) continue;

			if(cur_occ.at<uchar>(qx,qy) == 0){
				float spq = get_weight(I, px, py, qx, qy, shift_map);
				den += spq;
				num.val[0] += spq*I.at<Vec3f>(px + shift_map[qx][qy].first, py + shift_map[qx][qy].second).val[0];
				num.val[1] += spq*I.at<Vec3f>(px + shift_map[qx][qy].first, py + shift_map[qx][qy].second).val[1];
				num.val[2] += spq*I.at<Vec3f>(px + shift_map[qx][qy].first, py + shift_map[qx][qy].second).val[2];
			}
		}
	}

	num.val[0] /= den;
	num.val[1] /= den;
	num.val[2] /= den;

	I.at<Vec3f>(px, py).val[0] = num.val[0];
	I.at<Vec3f>(px, py).val[1] = num.val[1];
	I.at<Vec3f>(px, py).val[2] = num.val[2];
}

///////////////////////////////////////////////////////////////////

//Function to reconstruct one layer during the onion-peel initialization
//If there is no more layer to fill it returns false. Else, it returns true
bool one_peel(Mat &I_transf, Mat &cur_occ, vector< vector<ii> > &shift_map){
	vector<ii> peel;
	bool entry = false;

	for(int i=0;i<I_transf.rows;i++){
		for(int j=0;j<I_transf.cols;j++){
			if(is_border(cur_occ, i, j)){
				entry = true;
				pixel_reconstruction_init(I_transf, cur_occ, i, j, shift_map);
				peel.push_back({i, j});
			}
		}
	}

	for(int i=0;i<(int)peel.size();i++) cur_occ.at<uchar>(peel[i].first, peel[i].second) = 0;

	return entry;
}

///////////////////////////////////////////////////////////////////

//Function to do onion-peel initialization
void onion_peel(Mat &I_transf, Mat &cur_occ, vector< vector<ii> > &shift_map){
	while(one_peel(I_transf, cur_occ, shift_map)) continue;
}

///////////////////////////////////////////////////////////////////

//Function to calculate norm l1 between the previous solution and the current one.
//It's used to see if we get the cnvergence criteria 
float norm_l1(Mat &I_cur, Mat &I_cur_copy){
	float ans = 0.0;

	for(int i=0;i<I_cur.rows;i++){
		for(int j=0;j<I_cur.cols;j++){
			ans += abs(I_cur.at<Vec3f>(i,j).val[0] - I_cur_copy.at<Vec3f>(i,j).val[0]);
			ans += abs(I_cur.at<Vec3f>(i,j).val[1] - I_cur_copy.at<Vec3f>(i,j).val[1]);
			ans += abs(I_cur.at<Vec3f>(i,j).val[2] - I_cur_copy.at<Vec3f>(i,j).val[2]);
		}
	}

	return ans;
}

///////////////////////////////////////////////////////////////////

//Function to correct a occlusion to only have white and black pixels
//Because during the process of getting gaussian pyramid some pixels get gray between the white and the black ones
void correct_gray_area(Mat &occ_input){
	for(int i=0;i<occ_input.rows;i++){
		for(int j=0;j<occ_input.cols;j++){
			if(occ_input.at<uchar>(i,j) > 0 && occ_input.at<uchar>(i,j) <= 127) occ_input.at<uchar>(i,j) = 0;
			else if(occ_input.at<uchar>(i,j) > 127 && occ_input.at<uchar>(i,j) < 255) occ_input.at<uchar>(i,j) = 255;
		}
	}
}

///////////////////////////////////////////////////////////////////

//Function to get gradient components through an image
pair<Mat, Mat> get_gradient(Mat &I){
	Mat I_gray;
	cvtColor(I, I_gray, COLOR_BGR2GRAY);
	Mat Gx = Mat::zeros(I.rows, I.cols, CV_16SC1);
	Mat Gy = Mat::zeros(I.rows, I.cols, CV_16SC1);

	int GxKernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
	int GyKernel[3][3] = {{-1, -2 ,-1}, {0, 0, 0}, {1, 2, 1}};

	for(int i=0;i<I_gray.rows - 3;i++){
		for(int j=0;j<I_gray.cols - 3;j++){
			Mat window_imageX(I_gray, Rect(j, i, 3, 3));
			Mat window_imageY(I_gray, Rect(j, i, 3, 3));

			for(int k=0;k<window_imageX.rows;k++){
				for(int m=0;m<window_imageX.cols;m++){
					window_imageX.at<int>(k,m) *= GxKernel[k][m];
					window_imageY.at<int>(k,m) *= GyKernel[k][m];
				}
			}

			for(int k=0;k<window_imageX.rows;k++){
				for(int m=0;m<window_imageX.cols;m++){
					if(k == 1 && m == 1) continue;

					window_imageX.at<int>(1,1) += window_imageX.at<int>(k,m);
					window_imageY.at<int>(1,1) += window_imageY.at<int>(k,m);
				}
			}

			Gx.at<int>(i+1,j+1) = window_imageX.at<int>(1,1);
			Gy.at<int>(i+1,j+1) = window_imageY.at<int>(1,1);
		}
	}

	return {Gx, Gy};
}

///////////////////////////////////////////////////////////////////

//Function to get texture components through an image
pair<Mat, Mat> get_texture_components(Mat &Gx, Mat &Gy, int levels){
	Mat Tx = Mat::zeros(Gx.rows, Gx.cols, CV_32FC1);
	Mat Ty = Mat::zeros(Gx.rows, Gx.cols, CV_32FC1);
	int neighborhood_size = (1 << (levels-1));

	for(int row = 0; row < Gx.rows; row++){
		for(int col = 0; col < Gx.cols; col++){
			int card = 0;

			for(int i = -neighborhood_size/2; i <= neighborhood_size/2; i++){
				for(int j = -neighborhood_size/2; j <= neighborhood_size/2; j++){
					if(row+i < 0 || row+i >= Gx.rows || col+j < 0 || col+j >= Gx.cols) continue;

					card++;
					Tx.at<float>(row,col) += Gx.at<float>(row+i,col+j);
					Ty.at<float>(row,col) += Gy.at<float>(row+i,col+j);
				}
			}

			Tx.at<float>(row,col) /= card;
			Ty.at<float>(row,col) /= card;
		}
	}
	
	return {Tx, Ty};
}

///////////////////////////////////////////////////////////////////

int main(){
//Reading image and converting from 8UC3 to 32FC3
Mat I = imread("../barbara.png");
Mat I_transf = char_to_float(I);

//Reading the occlusion
Mat occ_input = imread("../barbara_occlusion.png");

//Getting the number of levels for the gaussian pyramid
int levels = get_levels(occ_input);

//Getting the occlusion gaussian pyramid
vector<Mat> occ_pyramid_color = get_pyramid(occ_input, levels);
vector<Mat> occ_pyramid;

//Converting the occlusion to have only one channel (8UC1)
for(int i=0;i<levels;i++){
	Mat occ_gray, occ_copy = occ_pyramid_color[i];
	cvtColor(occ_copy, occ_gray, COLOR_BGR2GRAY);
	occ_pyramid.push_back(occ_gray);
}

//Getting the gaussian pyramid for the image
vector<Mat> I_pyramid = get_pyramid(I_transf, levels);

//Declaring the shift maps for each level of the pyramid
vector<vector< vector<ii> > > shift_map_pyramid(levels);

//Correct the occlusion pyramid to only have white and black pixels
for(int i=0;i<levels;i++) correct_gray_area(occ_pyramid[i]);

//Doing random initialization of the shift map at the top of the pyramid 
vector< vector<ii> > shift_map(I.rows/((int)pow(2,levels-1)), vector<ii>(I.cols/((int)pow(2,levels-1))));
initialize_shift_map(occ_pyramid[levels-1], shift_map);

//Getting sigma at the top of the pyramid
sigma = get_sigma(I_pyramid[levels-1], shift_map);

Mat cur_occ = occ_pyramid[levels-1];

//Doing onion-peel initialization for the top level of the pyramid
onion_peel(I_pyramid[levels-1], cur_occ, shift_map);

shift_map_pyramid[levels-1] = shift_map;

//Begin of the algorithm
for(int l=levels-1;l>=0;l--){
	int k = 0;
	float e = 1.0;

	//Convergence criteria
	while(e > 0.1 && k < 10){
		Mat I_cur = I_pyramid[l];
		Mat occ_cur = occ_pyramid[l];
		int occ_cont = 0;

		patch_match(I_pyramid[l], occ_pyramid[l], shift_map_pyramid[l]);

		for(int i=0;i<occ_pyramid[l].rows;i++){
			for(int j=0;j<occ_pyramid[l].cols;j++){
				if(occ_pyramid[l].at<uchar>(i,j)== 255){
					pixel_reconstruction(I_pyramid[l], i, j, shift_map_pyramid[l]);
					occ_cont++;	
				} 
			}
		}

		e = norm_l1(I_pyramid[l], I_cur)/(3.0*occ_cont);
		k++;
	}

	//If we are at the base of the pyramid we do the final reconstruction
	if(l == 0){
		for(int i=0;i<occ_pyramid[l].rows;i++){
			for(int j=0;j<occ_pyramid[l].cols;j++){
				if(occ_pyramid[l].at<uchar>(i,j) == 255)
					final_reconstruction(I_pyramid[l], i, j, shift_map_pyramid[l]); 
			}
		}
	}
	else{
		//Else we upsample the shift map using nearest neighbor interpolation
		//And we get initial solution for the next level
		shift_map_pyramid[l-1] = get_upsampled_shift_map(shift_map_pyramid[l]);

		for(int i=0;i<occ_pyramid[l-1].rows;i++){
			for(int j=0;j<occ_pyramid[l-1].cols;j++){
				if(occ_pyramid[l-1].at<uchar>(i,j) == 255)
					pixel_reconstruction(I_pyramid[l-1], i, j, shift_map_pyramid[l-1]);
			}
		}
	} 
}

//Showing the result of the algorithm at the base of the pyramid
imshow("Final image", I_pyramid[0]);
waitKey(0);
return 0;
}