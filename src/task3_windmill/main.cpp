// =====================================================================
// 任务3：能量机关识别与稳定跟踪
// 处理小能量机关(task_3.mp4, 最多 1 个目标亮起)
// 和大能量机关(task_4.mp4, 最多 2 个目标亮起)视频。
//
// 识别流程（每帧）：
//   1. HSV 橙色掩膜 -> 检测 R 标（圆环 + 中心 R 字实心块）
//   2. 高亮掩膜(V>170) -> 亮起叶片元素（灯串 + 扇叶圆）
//   3. 按最近 R 标划分元素 -> 扇叶圆 = 最远端紧凑亮块(带内侧灯串)，
//      或整条亮起灯带的最外端(远视距/合并情形)
//   4. R 标跟踪：最近邻 + 全局位移预测（相机平移时跟随中心）
//   5. 目标跟踪：R 相对坐标关联，锁定 ID，丢失标记，
//      超时或目标熄灭后允许重选（详见 result/task3_tracking_result.md）
//
// 输出：result/task3_windmill/<task>/recognition_overlay.mp4
//       result/task3_windmill/<task>/binary_process.mp4
// =====================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// ---------------------------------------------------------------------
// 参数（依据两段视频统计标定）
// ---------------------------------------------------------------------
namespace cfg {
// 橙色掩膜
const int   H_LO = 0, H_HI = 35, S_LO = 30, V_LO = 30, V_HI = 255;
const int   BRIGHT_V = 170;          // 亮起灯条的 V 下限
// R 标圆环
const int   RING_MIN_SIZE = 40, RING_MAX_SIZE = 160;
const double RING_FILL_LO = 0.25, RING_FILL_HI = 0.70;
const int   RING_MIN_AREA = 100;
// R 字实心块
const int   DISC_MAX_AREA = 900, DISC_MAX_SIZE = 34;
const double DISC_FILL = 0.5;
const double DISC_MAX_DIST = 0.55;   // 距圆环中心(相对环尺寸)
const double DISC_SIZE_LO = 0.12, DISC_SIZE_HI = 0.50;
// R 标跟踪
const double R_MATCH_DIST = 150.0;   // 检测-跟踪最近邻上限
const double R_NEW_GUARD  = 60.0;    // 新跟踪与已有跟踪的距离护栏
const int    R_CONFIRM    = 3;       // 连续帧数确认后才算有效 R 标
const int    R_DROP_LOST  = 90;      // R 标丢失多少帧后移除
// 亮元素
const int    ELEM_MIN_AREA = 40;
const double GATE_LO = 0.30, GATE_HI = 4.60;  // 相对环尺寸的环带(叶片远端可到 4.5x)
// 目标（扇叶圆）
const double TIP_FILL = 0.40, TIP_ASPECT = 1.9;
const double TIP_SIZE_LO = 0.18, TIP_SIZE_HI = 0.55;
const double CHAIN_FILL = 0.70, CHAIN_DIST = 1.6;  // 灯串:低填充,距扇叶圆近
const double TIP_MIN_D  = 0.90;      // 扇叶圆到 R 的最小距离(相对环)
const double MERGED_SIZE = 0.80, MERGED_ASPECT = 2.0;  // 合并灯带情形
// 目标槽位（锁定/丢失/重选）
const double SLOT_R_BIND    = 150.0; // 槽位与 R 标绑定的距离
const double SLOT_REL_MATCH = 45.0;  // R 相对坐标关联阈值
const double SLOT_REL_RATIO = 0.65;  // 关联阈值随环尺寸放宽
const int    LOST_TIMEOUT   = 20;    // 持续丢失多少帧后允许重选
const int    GRAVE_KEEP     = 120;   // 过期槽位保留(可同 ID 复活)
const double NEW_SLOT_GUARD = 200.0; // 已有存活槽位附近不再新建
const double GS_DECAY       = 0.85;  // 无匹配时全局位移衰减
}  // namespace cfg

// ---------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------
struct Elem {
    Point2f c;      // 质心
    float   size;   // max(w,h)
    float   fill;   // area / (w*h)
    float   aspect; // max(w,h)/min(w,h)
    float   area;
    Rect    box;
};

struct RCenter {
    Point2f pos;
    float   size;
};

struct RTrack {
    Point2f pos;
    float   size;
    int     seen = 0;
    int     lost = 0;
    // 目标确认(连续 2 帧有效检测才建立槽位,滤除单帧闪烁)
    Point2f pend_rel;
    int     pend_count = 0;
};

struct Slot {
    int     id;
    Point2f rpos;    // 绑定的 R 标位置(屏坐标)
    float   rsize;   // 绑定时 R 环尺寸
    Point2f rel;     // 目标相对 R 标的位置(R 相对坐标,EMA 平滑)
    Point2f meas;    // 本帧观测 rel(未匹配时无效)
    bool    matched = false;
    int     lost = 0;
    int     total_lost = 0;
};

// ---------------------------------------------------------------------
// 橙色掩膜
// ---------------------------------------------------------------------
static Mat orangeMask(const Mat& frame) {
    Mat hsv, mask;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(cfg::H_LO, cfg::S_LO, cfg::V_LO),
            Scalar(cfg::H_HI, 255, cfg::V_HI), mask);
    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(mask, mask, MORPH_CLOSE, k);
    return mask;
}

// ---------------------------------------------------------------------
// 检测 R 标（圆环 + 中心 R 字实心块）
// ---------------------------------------------------------------------
static vector<RCenter> detectRCenters(const Mat& omask) {
    Mat labels, stats, cent;
    int n = connectedComponentsWithStats(omask, labels, stats, cent);

    struct Cand { Point2f c; float size; };
    vector<Cand> rings, discs;

    for (int i = 1; i < n; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        if (w <= 0 || h <= 0) continue;
        double fill = (double)area / (w * h);
        int sz = max(w, h);

        if (area >= cfg::RING_MIN_AREA && min(w, h) >= cfg::RING_MIN_SIZE &&
            sz <= cfg::RING_MAX_SIZE && fill >= cfg::RING_FILL_LO &&
            fill <= cfg::RING_FILL_HI) {
            // 圆环需存在内轮廓(空洞)
            Mat comp = (labels == i);
            vector<vector<Point>> cnts;
            vector<Vec4i> hier;
            findContours(comp, cnts, hier, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            bool hole = false;
            for (auto& hh : hier) {
                if (hh[3] != -1) { hole = true; break; }
            }
            if (hole) rings.push_back({Point2f(cent.at<double>(i,0),
                                               cent.at<double>(i,1)), (float)sz});
        } else if (area < cfg::DISC_MAX_AREA && fill >= cfg::DISC_FILL &&
                   sz <= cfg::DISC_MAX_SIZE) {
            discs.push_back({Point2f(cent.at<double>(i,0),
                                     cent.at<double>(i,1)), (float)sz});
        }
    }

    vector<RCenter> out;
    for (auto& r : rings) {
        for (auto& d : discs) {
            double dist = norm(r.c - d.c);
            if (dist < cfg::DISC_MAX_DIST * r.size &&
                d.size >= cfg::DISC_SIZE_LO * r.size &&
                d.size <= cfg::DISC_SIZE_HI * r.size) {
                out.push_back({r.c, r.size});
                break;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------
// 高亮元素（亮起灯条/扇叶圆）
// ---------------------------------------------------------------------
static vector<Elem> detectBrightElems(const Mat& omask, const Mat& v_ch) {
    Mat bmask;
    threshold(v_ch, bmask, cfg::BRIGHT_V, 255, THRESH_BINARY);
    bitwise_and(omask, bmask, bmask);
    Mat k = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(bmask, bmask, MORPH_CLOSE, k);  // 合并灯串小点

    Mat labels, stats, cent;
    int n = connectedComponentsWithStats(bmask, labels, stats, cent);
    vector<Elem> elems;
    for (int i = 1; i < n; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < cfg::ELEM_MIN_AREA) continue;
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        if (w <= 0 || h <= 0) continue;
        Elem e;
        e.c = Point2f((float)cent.at<double>(i,0), (float)cent.at<double>(i,1));
        e.size = (float)max(w, h);
        e.fill = (float)area / (w * h);
        e.aspect = (float)max(w, h) / (float)max(1, min(w, h));
        e.area = (float)area;
        e.box = Rect(x, y, w, h);
        elems.push_back(e);
    }
    return elems;
}

// ---------------------------------------------------------------------
// 由 R 标周围的亮元素解算目标扇叶圆
// 返回 true 时 tip 为扇叶圆中心,size 为其尺寸
// ---------------------------------------------------------------------
static bool targetFromElems(const Point2f& r, float rsize,
                            const vector<Elem>& elems,
                            Point2f& tip, float& size) {
    auto dR = [&](const Elem& e) { return norm(e.c - r); };

    // 环带过滤
    vector<Elem> gated;
    for (auto& e : elems) {
        double d = dR(e);
        if (d >= cfg::GATE_LO * rsize && d <= cfg::GATE_HI * rsize)
            gated.push_back(e);
    }
    if (gated.empty()) return false;

    // 装甲灯条对排除：细长元素成对出现(两条平行竖条)是场地装甲灯条，
    // 不是叶片灯串。灯串是单一弧带。
    auto is_strip_pair = [&](const Elem& e) {
        for (auto& o : gated) {
            if (o.aspect < 3.5) continue;
            if (norm(o.c - e.c) > 1.2f * rsize) continue;
            if (norm(o.c - e.c) < 0.2f * rsize) continue;
            return true;
        }
        return false;
    };

    // 候选扇叶圆：紧凑高填充亮块
    vector<Elem> tips;
    for (auto& e : gated) {
        if (e.fill >= cfg::TIP_FILL && e.aspect < cfg::TIP_ASPECT &&
            e.size >= cfg::TIP_SIZE_LO * rsize &&
            e.size <= cfg::TIP_SIZE_HI * rsize)
            tips.push_back(e);
    }
    if (!tips.empty()) {
        // 扇叶圆 = 距 R 最远的候选
        Elem tip_e = *max_element(tips.begin(), tips.end(),
                                  [&](const Elem& a, const Elem& b) {
                                      return dR(a) < dR(b);
                                  });
        double tip_d = dR(tip_e);
        if (tip_d < cfg::TIP_MIN_D * rsize) return false;
        // 需要内侧灯串：低填充元素，位于扇叶圆与 R 之间且在扇叶圆附近
        bool has_chain = false;
        for (auto& e : gated) {
            if (e.fill > cfg::CHAIN_FILL) continue;
            if (dR(e) >= tip_d - 0.05 * rsize) continue;
            if (norm(e.c - tip_e.c) > cfg::CHAIN_DIST * rsize) continue;
            if (is_strip_pair(e)) continue;  // 装甲灯条对
            has_chain = true;
            break;
        }
        if (!has_chain) return false;
        tip = tip_e.c;
        size = tip_e.size;
        return true;
    }

    // 无独立扇叶圆：整条亮起灯带(远视距/形态学合并)，取离 R 最远端
    Elem far_e = *max_element(gated.begin(), gated.end(),
                              [&](const Elem& a, const Elem& b) {
                                  return dR(a) < dR(b);
                              });
    double far_d = dR(far_e);
    if (far_d < cfg::TIP_MIN_D * rsize) return false;
    if (far_e.size >= cfg::MERGED_SIZE * rsize &&
        far_e.aspect >= cfg::MERGED_ASPECT && far_e.fill <= cfg::CHAIN_FILL &&
        !is_strip_pair(far_e)) {
        // 取包围盒中离 R 最远的角点
        Point2f c1(far_e.box.x, far_e.box.y);
        Point2f c2(far_e.box.x + far_e.box.width, far_e.box.y);
        Point2f c3(far_e.box.x, far_e.box.y + far_e.box.height);
        Point2f c4(far_e.box.x + far_e.box.width,
                   far_e.box.y + far_e.box.height);
        Point2f far_pt = c1;
        double bd = norm(c1 - r);
        for (auto& p : {c2, c3, c4}) {
            double d = norm(p - r);
            if (d > bd) { bd = d; far_pt = p; }
        }
        tip = far_pt;
        size = far_e.size;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_path>\n";
        return -1;
    }
    string video_path = argv[1];
    cout << "Input video: " << video_path << "\n";

    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Cannot open video: " << video_path << "\n";
        return -1;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);
    int frame_count = (int)cap.get(CAP_PROP_FRAME_COUNT);
    if (fps <= 0) fps = 30.0;

    string out_dir;
    if (video_path.find("task_3") != string::npos)
        out_dir = "result/task3_windmill/task_3/";
    else if (video_path.find("task_4") != string::npos)
        out_dir = "result/task3_windmill/task_4/";
    else
        out_dir = "result/task3_windmill/unknown/";

    VideoWriter writer(out_dir + "recognition_overlay.mp4",
                       VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height));
    VideoWriter bin_writer(out_dir + "binary_process.mp4",
                           VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height));
    if (!writer.isOpened() || !bin_writer.isOpened()) {
        cerr << "Cannot create output video under " << out_dir << "\n";
        return -1;
    }

    // ---- 跟踪状态 ----
    vector<RTrack> r_tracks;
    vector<Slot> slots, grave;
    int next_id = 1;
    Point2f gs(0, 0);  // 全局位移(相机运动)持久化预测

    Mat frame;
    int frame_id = 0;
    int total_lost_frames = 0;

    while (cap.read(frame)) {
        // ---------- 1. 掩膜与检测 ----------
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        vector<Mat> hsv_ch;
        split(hsv, hsv_ch);
        Mat omask = orangeMask(frame);
        vector<RCenter> r_det = detectRCenters(omask);
        vector<Elem> elems = detectBrightElems(omask, hsv_ch[2]);

        // ---------- 2. R 标跟踪(最近邻 + 全局位移) ----------
        // 2a. 检测 -> 跟踪最近邻匹配
        vector<pair<int, RCenter>> matched;  // 跟踪下标 -> 检测
        vector<bool> det_used(r_det.size(), false);
        for (size_t ti = 0; ti < r_tracks.size(); ti++) {
            double bd = 1e18;
            int bi = -1;
            for (size_t di = 0; di < r_det.size(); di++) {
                if (det_used[di]) continue;
                double d = norm(r_det[di].pos - r_tracks[ti].pos);
                if (d < bd) { bd = d; bi = (int)di; }
            }
            if (bi >= 0 && bd <= cfg::R_MATCH_DIST) {
                matched.push_back({(int)ti, r_det[bi]});
                det_used[bi] = true;
            }
        }

        // 2b. 全局位移 = 匹配对位移中值；无匹配则按衰减持久化
        vector<double> dx, dy;
        for (auto& m : matched) {
            dx.push_back(m.second.pos.x - r_tracks[m.first].pos.x);
            dy.push_back(m.second.pos.y - r_tracks[m.first].pos.y);
        }
        if (!dx.empty()) {
            sort(dx.begin(), dx.end());
            sort(dy.begin(), dy.end());
            gs = Point2f((float)dx[dx.size()/2], (float)dy[dy.size()/2]);
        } else {
            gs *= (float)cfg::GS_DECAY;
        }

        // 2c. 更新跟踪
        for (size_t ti = 0; ti < r_tracks.size(); ti++) {
            auto it = find_if(matched.begin(), matched.end(),
                              [&](const pair<int, RCenter>& m) {
                                  return m.first == (int)ti;
                              });
            if (it != matched.end()) {
                r_tracks[ti].pos = it->second.pos;
                r_tracks[ti].size = 0.7f * it->second.size + 0.3f * r_tracks[ti].size;
                r_tracks[ti].seen++;
                r_tracks[ti].lost = 0;
            } else {
                r_tracks[ti].pos += gs;  // 相机运动预测
                r_tracks[ti].lost++;
            }
        }
        r_tracks.erase(remove_if(r_tracks.begin(), r_tracks.end(),
                                 [](const RTrack& t) {
                                     return t.lost >= cfg::R_DROP_LOST;
                                 }),
                       r_tracks.end());

        // 2d. 未匹配检测 -> 新跟踪(带距离护栏防重复)
        for (size_t di = 0; di < r_det.size(); di++) {
            if (det_used[di]) continue;
            bool near_exist = false;
            for (auto& t : r_tracks) {
                if (norm(r_det[di].pos - t.pos) <= cfg::R_NEW_GUARD) {
                    near_exist = true;
                    break;
                }
            }
            if (!near_exist) {
                RTrack t;
                t.pos = r_det[di].pos;
                t.size = r_det[di].size;
                t.seen = 1;
                r_tracks.push_back(t);
            }
        }

        // 槽位绑定的 R 位置随全局位移更新
        for (auto& s : slots) s.rpos += gs;
        for (auto& s : grave) s.rpos += gs;

        // ---------- 3. 目标检测(按已确认 R 标划分元素) ----------
        struct Det { int r_track; Point2f tip; float size; };
        vector<Det> dets;
        for (size_t ti = 0; ti < r_tracks.size(); ti++) {
            if (r_tracks[ti].seen < cfg::R_CONFIRM) continue;
            // 元素划分到最近的已确认 R 标
            vector<Elem> mine;
            for (auto& e : elems) {
                int best = -1;
                double bd = 1e18;
                for (size_t tj = 0; tj < r_tracks.size(); tj++) {
                    if (r_tracks[tj].seen < cfg::R_CONFIRM) continue;
                    double d = norm(e.c - r_tracks[tj].pos);
                    if (d < bd) { bd = d; best = (int)tj; }
                }
                if (best == (int)ti) mine.push_back(e);
            }
            Point2f tip;
            float tsize;
            if (targetFromElems(r_tracks[ti].pos, r_tracks[ti].size,
                                mine, tip, tsize)) {
                dets.push_back({(int)ti, tip, tsize});
            }
        }

        // ---------- 4. 槽位关联(锁定/丢失/重选) ----------
        for (auto& s : slots) s.matched = false;
        for (auto& s : grave) s.matched = false;

        for (auto& d : dets) {
            Point2f rel = d.tip - r_tracks[d.r_track].pos;

            // 找同一 R 标附近的槽位(存活 + 过期均可复活)
            Slot* cand = nullptr;
            double cd = 1e18;
            for (auto& s : slots) {
                double dist = norm(s.rpos - r_tracks[d.r_track].pos);
                if (dist <= cfg::SLOT_R_BIND && dist < cd) {
                    cd = dist;
                    cand = &s;
                }
            }
            if (cand == nullptr) {
                for (auto& s : grave) {
                    double dist = norm(s.rpos - r_tracks[d.r_track].pos);
                    if (dist <= cfg::SLOT_R_BIND && dist < cd) {
                        cd = dist;
                        cand = &s;
                    }
                }
            }

            if (cand != nullptr) {
                if (!cand->matched) {
                    double drel = norm(rel - cand->rel);
                    if (drel <= max(cfg::SLOT_REL_MATCH,
                                    cfg::SLOT_REL_RATIO * cand->rsize)) {
                        cand->meas = rel;
                        cand->matched = true;
                        cand->rpos = r_tracks[d.r_track].pos;
                        // 从过期区复活(同 ID 保持身份)
                        auto it = find_if(grave.begin(), grave.end(),
                                          [&](const Slot& s) {
                                              return s.id == cand->id;
                                          });
                        if (it != grave.end()) {
                            slots.push_back(*it);
                            grave.erase(it);
                        }
                    }
                }
            } else {
                // 已有存活槽位在附近(重复 R 标保护)则跳过
                bool near_alive = false;
                for (auto& s : slots) {
                    if (norm(s.rpos - r_tracks[d.r_track].pos) <= cfg::NEW_SLOT_GUARD) {
                        near_alive = true;
                        break;
                    }
                }
                if (!near_alive) {
                    // 目标确认：同一 R 标连续 2 帧检出才建立槽位
                    RTrack& tr = r_tracks[d.r_track];
                    double pd = norm(rel - tr.pend_rel);
                    if (tr.pend_count > 0 &&
                        pd <= max(cfg::SLOT_REL_MATCH,
                                  cfg::SLOT_REL_RATIO * tr.size)) {
                        tr.pend_count++;
                    } else {
                        tr.pend_rel = rel;
                        tr.pend_count = 1;
                    }
                    if (tr.pend_count >= 2) {
                        Slot s;
                        s.id = next_id++;
                        s.rpos = tr.pos;
                        s.rsize = tr.size;
                        s.rel = rel;
                        s.meas = rel;
                        s.matched = true;
                        s.lost = 0;
                        slots.push_back(s);
                        tr.pend_count = 0;
                    }
                }
            }
        }

        // 槽位状态更新
        for (auto it = slots.begin(); it != slots.end();) {
            if (it->matched) {
                it->rel = 0.8f * it->meas + 0.2f * it->rel;
                it->lost = 0;
                ++it;
            } else {
                it->lost++;
                it->total_lost++;
                total_lost_frames++;
                if (it->lost >= cfg::LOST_TIMEOUT) {
                    grave.push_back(*it);
                    it = slots.erase(it);
                } else {
                    ++it;
                }
            }
        }
        // 无检测时清空待确认计数
        for (auto& tr : r_tracks) {
            bool has_det = false;
            for (auto& d : dets) {
                if (d.r_track == (int)(&tr - &r_tracks[0])) { has_det = true; break; }
            }
            if (!has_det) tr.pend_count = 0;
        }        grave.erase(remove_if(grave.begin(), grave.end(),
                              [](const Slot& s) {
                                  return s.lost >= cfg::GRAVE_KEEP;
                              }),
                    grave.end());

        // ---------- 5. 可视化 ----------
        Mat overlay = frame.clone();
        // 5a. R 标中心
        for (auto& t : r_tracks) {
            if (t.seen < cfg::R_CONFIRM) continue;
            circle(overlay, t.pos, (int)(t.size / 2), Scalar(0, 200, 0), 2);
            line(overlay, t.pos - Point2f(8, 0), t.pos + Point2f(8, 0),
                 Scalar(0, 255, 0), 2);
            line(overlay, t.pos - Point2f(0, 8), t.pos + Point2f(0, 8),
                 Scalar(0, 255, 0), 2);
            putText(overlay, "R", t.pos + Point2f(t.size/2 + 4, 4),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        // 5b. 目标(含丢失状态)
        vector<Scalar> colors = {
            Scalar(255, 255, 0),   // 青
            Scalar(255, 0, 255),   // 品红
            Scalar(0, 165, 255),   // 橙
            Scalar(255, 255, 255),
        };
        for (auto& s : slots) {
            Scalar col = colors[(s.id - 1) % colors.size()];
            Point2f tip = s.rpos + s.rel;
            float draw_r = max(8.0f, 0.5f * s.rsize * 0.25f + 6.0f);

            if (s.matched) {
                // 扇叶圆轮廓 + 中心 + 连线
                circle(overlay, tip, (int)draw_r, col, 2);
                circle(overlay, tip, 3, col, -1);
                line(overlay, s.rpos, tip, col, 2);
            } else {
                // 丢失：虚线效果 + 更细的标记
                circle(overlay, tip, (int)draw_r, Scalar(0, 0, 255), 1);
                line(overlay, s.rpos, tip, Scalar(0, 0, 255), 1);
                circle(overlay, tip, 3, Scalar(0, 0, 255), -1);
            }

            double angle = atan2(s.rel.y, s.rel.x) * 180.0 / CV_PI;
            if (angle < 0) angle += 360.0;
            string status = s.matched ? "detected" : "lost";
            string text = "T" + to_string(s.id) + " " + status + " " +
                          cv::format("%.0f deg", angle);
            putText(overlay, text,
                    tip + Point2f(draw_r + 6, 4),
                    FONT_HERSHEY_SIMPLEX, 0.55, s.matched ? col : Scalar(0, 0, 255), 2);
        }

        // 5c. 左上角状态汇总
        string summary = "targets: " + to_string(slots.size());
        for (auto& s : slots) {
            summary += "  T" + to_string(s.id) + (s.matched ? ":det" : ":lost");
        }
        if (slots.empty()) summary += "  (no target)";
        putText(overlay, summary, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        // 5d. 二值化过程图(2x2: 原图 | 橙色掩膜 | 高亮掩膜 | 叠加结果)
        Mat bmask;
        threshold(hsv_ch[2], bmask, cfg::BRIGHT_V, 255, THRESH_BINARY);
        bitwise_and(omask, bmask, bmask);
        Mat bin_frame(height, width, CV_8UC3, Scalar(0, 0, 0));
        Mat hw(height / 2, width / 2, CV_8UC1);
        auto put_pane = [&](const Mat& src, int px, int py) {
            Mat dst = bin_frame(Rect(px * width / 2, py * height / 2,
                                     width / 2, height / 2));
            if (src.channels() == 1) {
                Mat c3;
                cvtColor(src, c3, COLOR_GRAY2BGR);
                resize(c3, dst, dst.size());
            } else {
                resize(src, dst, dst.size());
            }
        };
        put_pane(frame, 0, 0);
        put_pane(omask, 1, 0);
        put_pane(bmask, 0, 1);
        put_pane(overlay, 1, 1);
        putText(bin_frame, "original", Point(10, height/2 - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
        putText(bin_frame, "orange mask", Point(width/2 + 10, height/2 - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
        putText(bin_frame, "bright mask (V>170)", Point(10, height - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);
        putText(bin_frame, "overlay", Point(width/2 + 10, height - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);

        writer.write(overlay);
        bin_writer.write(bin_frame);

        // ---------- 6. 日志 ----------
        if (frame_id % 30 == 0) {
            cout << "\n----- frame " << frame_id << " : R tracks="
                 << r_tracks.size() << " slots=" << slots.size() << " -----\n";
            for (auto& t : r_tracks) {
                if (t.seen >= cfg::R_CONFIRM)
                    cout << "  R(" << (int)t.pos.x << "," << (int)t.pos.y
                         << ") size=" << (int)t.size << "\n";
            }
            for (auto& s : slots) {
                Point2f tip = s.rpos + s.rel;
                cout << "  T" << s.id << " " << (s.matched ? "detected" : "lost")
                     << " tip=(" << (int)tip.x << "," << (int)tip.y
                     << ") lost=" << s.lost << "\n";
            }
        }

        frame_id++;
        cout << "\rProcessing frame: " << frame_id << " / " << frame_count;
        cout.flush();
    }

    cap.release();
    writer.release();
    bin_writer.release();

    cout << "\n\nFinished! frames=" << frame_id
         << " max_id=" << (next_id - 1)
         << " lost_frames=" << total_lost_frames << "\n";
    cout << "Output:\n  " << out_dir << "recognition_overlay.mp4\n  "
         << out_dir << "binary_process.mp4\n";
    return 0;
}
