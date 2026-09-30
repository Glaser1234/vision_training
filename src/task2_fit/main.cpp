#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>


using namespace cv;
using namespace std;
struct ThetaResidual{
    ThetaResidual(double t,double theta):t_(t),theta_(theta){}
    template<typename T>
    bool operator()
    (
    const T* A,
    const T* b,
    const T* Omega,
    const T* phi,
    const T* theta0 , T* residual
    )const{
        T theta_model = theta0[0]+b[0]*T(t_)+A[0]/Omega[0]*(cos(phi[0])-cos(Omega[0]*T(t_)+phi[0]));
    
    residual[0]=T(theta_)-theta_model;
    return true;
    }
    double t_;
    double theta_;
};


// ===========================
// 数据结构
// ===========================

struct BallData
{
    int frame_id;

    double t;

    Point2f center;

    double theta_wrapped;

    double theta_unwrapped;
};



// ===========================
// 鲁棒 unwrap
// ===========================

double robustUnwrap(
    double theta_wrapped,
    double theta_predict
)
{

    double k =
        round(
            (theta_predict - theta_wrapped)
            /
            (2.0 * CV_PI)
        );


    return theta_wrapped
           +
           k * 2.0 * CV_PI;
}




int main()
{


    //============================
    // 打开视频
    //============================

    VideoCapture cap(
        "resources/task_2.mp4"
    );


    if(!cap.isOpened())
    {
        cerr<<"Cannot open video!"
            <<endl;

        return -1;
    }




    double fps = 60.0;

    VideoWriter writer(
        "result/task2_fit/tracking_overlay.mp4",
        VideoWriter::fourcc(
            'm','p','4','v'
        ),
        fps,
        Size(960,720)
    );

    if(!writer.isOpened())
    {
        cerr<<"Cannot open writer!"
            <<endl;
        return -1;
    }

    //============================
    // 参数
    //============================
    Point2f rotation_center(480,360);
    vector<BallData> data;
    int frame_id = 0;

    //============================
    // unwrap 状态
    //============================
    bool first_angle = true;
    double theta_prev = 0.0;
    double t_prev = 0.0;

    // 当前角速度估计
    double omega_est = 0.0;

    Mat kernel =
        getStructuringElement(
            MORPH_ELLIPSE,
            Size(3,3)
        );



    Mat frame;




    //============================
    // 主循环
    //============================


    while(cap.read(frame))
    {


        Mat hsv;


        cvtColor(
            frame,
            hsv,
            COLOR_BGR2HSV
        );



        Mat mask;


        // 青色阈值

        inRange(
            hsv,
            Scalar(
                80,
                60,
                60
            ),
            Scalar(
                100,
                255,
                255
            ),
            mask
        );



        morphologyEx(
            mask,
            mask,
            MORPH_OPEN,
            kernel
        );



        vector<vector<Point>> contours;


        vector<Vec4i> hierarchy;



        findContours(
            mask,
            contours,
            hierarchy,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE
        );




        int max_index = -1;


        double max_area = 0;



        for(int i=0;i<(int)contours.size();i++)
        {

            double area =
                contourArea(
                    contours[i]
                );


            if(area > max_area)
            {
                max_area = area;
                max_index = i;
            }

        }





        if(max_index!=-1)
        {


            Moments M =
                moments(
                    contours[max_index]
                );



            if(M.m00!=0)
            {


                Point2f center;


                center.x =
                    M.m10/M.m00;


                center.y =
                    M.m01/M.m00;




                //---------------------
                // 时间
                //---------------------

                double t =
                    frame_id / fps;





                //---------------------
                // atan2角度
                //---------------------

                double theta_wrapped =
                    atan2(
                        rotation_center.y-center.y,
                        center.x-rotation_center.x
                    );





                //---------------------
                // 鲁棒 unwrap
                //---------------------

                double theta_unwrapped;



                if(first_angle)
                {


                    theta_unwrapped =
                        theta_wrapped;


                    theta_prev =
                        theta_unwrapped;


                    t_prev =
                        t;


                    first_angle=false;

                }
                else
                {


                    // 预测角度

                    double theta_predict =
                        theta_prev
                        +
                        omega_est
                        *
                        (t-t_prev);



                    // 找最近的2π分支

                    theta_unwrapped =
                        robustUnwrap(
                            theta_wrapped,
                            theta_predict
                        );




                    double dt =
                        t-t_prev;



                    if(dt>0)
                    {


                        double omega_current =
                            (
                                theta_unwrapped
                                -
                                theta_prev
                            )
                            /
                            dt;



                        // 低通滤波

                        omega_est =
                            0.8*omega_est
                            +
                            0.2*omega_current;

                    }




                    theta_prev =
                        theta_unwrapped;


                    t_prev =
                        t;

                }






                //---------------------
                // 保存
                //---------------------

                BallData p;


                p.frame_id =
                    frame_id;


                p.t =
                    t;


                p.center =
                    center;


                p.theta_wrapped =
                    theta_wrapped;


                p.theta_unwrapped =
                    theta_unwrapped;



                data.push_back(p);






                circle(
                    frame,
                    center,
                    5,
                    Scalar(
                        0,
                        0,
                        255
                    ),
                    -1
                );



                putText(
                    frame,
                    "theta="
                    +
                    to_string(theta_unwrapped),
                    Point(
                        30,
                        40
                    ),
                    FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Scalar(
                        0,
                        255,
                        0
                    ),
                    2
                );

            }

        }





        // 显示圆心

        circle(
            frame,
            rotation_center,
            5,
            Scalar(
                255,
                0,
                0
            ),
            -1
        );





        writer.write(frame);



        imshow(
            "tracking",
            frame
        );


        imshow(
            "mask",
            mask
        );




        frame_id++;




        if(waitKey(1000/fps)==27)
        {
            break;
        }

    }


    cap.release();
    writer.release();

    destroyAllWindows();





    //============================
    // 输出CSV
    //============================


    ofstream csv(
        "result/task2_fit/data.csv"
    );



    csv
    <<"frame_id,"
    <<"time,"
    <<"x,"
    <<"y,"
    <<"theta_wrapped,"
    <<"theta_unwrapped\n";



    for(auto &p:data)
    {

        csv
        <<p.frame_id<<","
        <<p.t<<","
        <<p.center.x<<","
        <<p.center.y<<","
        <<p.theta_wrapped<<","
        <<p.theta_unwrapped
        <<"\n";
    }

    csv.close();



    //自动初始化参数
    double theta0 = data.front().theta_unwrapped;
    double b = (data.back().theta_unwrapped-data.front().theta_unwrapped)/(data.back().t-data.front().t);
    vector<double> omega;
    for(int i=1;i<data.size();i++){
        double w = (data[i].theta_unwrapped-data[i-1].theta_unwrapped)/(data[i].t-data[i-1].t);
        omega.push_back(w);
    }

    double omega_max = *max_element(omega.begin(),omega.end());
    double omega_min = *min_element(omega.begin(),omega.end());
    double A = (omega_max-omega_min)/2.0;
    if(A<0.01)A=0.5;
    double Omega = 2*CV_PI/4.0;
    double phi = 0.0;

    //ceres优化
    ceres::Problem problem;
    for(int i=0;i<data.size();i++){
        ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<ThetaResidual,1,1,1,1,1,1>(new ThetaResidual(data[i].t,data[i].theta_unwrapped));
        problem.AddResidualBlock(cost,new ceres::HuberLoss(0.1),&A,&b,&Omega,&phi,&theta0);
    }

    //参数约束
    problem.SetParameterLowerBound(&A,0,0.0);
    problem.SetParameterLowerBound(&Omega,0,0.1);
    
    //求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 200;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);

    //计算拟合角度,RMSE
    vector<double> theta_fit;

    for(auto &p:data){
        double theta_model = theta0+b*p.t+A/Omega*(cos(phi)-cos(Omega*p.t+phi));
        theta_fit.push_back(theta_model);
    }

    double mse = 0.0;
    for(int i=0;i<data.size();i++){
        double error = data[i].theta_unwrapped-theta_fit[i];
        mse += error*error;
    }
    double rmse = sqrt(mse/data.size());

    //输出拟合数据
    ofstream fit_csv("result/task2_fit/fit.csv");
    fit_csv<<"time,"<<"theta_data,"<<"theta_fit,"<<"error\n";
    for(int i=0;i<data.size();i++){
        double error = data[i].theta_unwrapped-theta_fit[i];
        fit_csv<<data[i].t<<","<<data[i].theta_unwrapped<<","<<theta_fit[i]<<","<<error<<"\n";
    }
    fit_csv.close();
    //输出角速度拟合数据
    ofstream omega_csv("result/task2_fit/omega_fit.csv");
    omega_csv<<"time,"<<"omega_data,"<<"omega_fit,"<<"error\n";
    double omega_mse = 0.0;
    for(int i=1;i<data.size();i++){
        double dt = data[i].t-data[i-1].t;
        double omega_data = (data[i].theta_unwrapped-data[i-1].theta_unwrapped)/dt;
        double omega_fit = b+A*sin(Omega*data[i].t+phi);
        double error = omega_data-omega_fit;
        omega_mse += error*error;

        omega_csv<<data[i].t<<","<<omega_data<<","<<omega_fit<<","<<error<<"\n";
    }
    omega_csv.close();

    double omega_rmse = sqrt(omega_mse/(data.size()-1));



    //输出最终拟合参数和RMSE
    cout<<summary.BriefReport()<<endl;
    while(phi>=CV_PI)phi-=2*CV_PI;
    while(phi<-CV_PI)phi+=2*CV_PI;
    cout<<"========== Result =========="<<endl;
    cout<<"A = "<<A<<endl;
    cout<<"b = "<<b<<endl;
    cout<<"Omega = "<<Omega<<endl;
    cout<<"phi = "<<phi<<endl;
    cout<<"theta0 = "<<theta0<<endl;
    cout<<"RMSE = "<<rmse<<" rad"<<endl;
    cout<<"Detected points: "<<data.size()<<endl;
    cout<<"Time range:"<<data.front().t<<"~"<<data.back().t<<" s"<<endl;
    cout<<"Omega RMSE = "<<omega_rmse<<" rad/s"<<endl;
    if(b<=A){
        cout<<"Warning: b <= A"<<endl;
    }


    // 简单检查
    for(int i=0;i<min(10,(int)data.size());i++){
        cout<<"frame="<<data[i].frame_id<<" t="<<data[i].t<<" theta="<<data[i].theta_unwrapped<<endl;
    }
    return 0;
}