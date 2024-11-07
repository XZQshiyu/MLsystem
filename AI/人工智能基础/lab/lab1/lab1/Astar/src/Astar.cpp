#include <vector>
#include <iostream>
#include <queue>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

using namespace std;

struct Map_Cell
{
    // 0: 可通行地块
    // 1: 不可通行地块
    // 2: 补给站
    // 3: 起点
    // 4: 终点
    int type;

    // TODO: 定义地图信息
};

struct Search_Cell
{
    int h;
    int g;
    int x, y;            // location
    Search_Cell *parent; // 父节点位置，追踪路径
    int supply;          // 剩余物资
    // pair<int, int> direction;   // 移动方向

    Search_Cell(int x, int y, int g, int h, int supply, Search_Cell *parent = nullptr)
        : x(x), y(y), g(g), h(h), supply(supply), parent(parent) {}

    int f() const { return g + h; } // compute f(n) = g(n) + h(n) 评价函数
    // TODO: 定义搜索状态
};

// 自定义比较函数对象，按照 Search_Cell 结构体的 g + h 属性进行比较
struct CompareF
{
    bool operator()(const Search_Cell *a, const Search_Cell *b) const
    {
        if (a->f() == b->f())
            return a->supply < b->supply;
        return a->f() > b->f();
        // return (a->g + a->h) > (b->g + b->h); // 较小的 g + h 值优先级更高
    }
};

// TODO: 定义启发式函数
int Heuristic_Funtion(int x1, int y1, int x2, int y2)
{
    // 曼哈顿距离
    return abs(x1 - x2) + abs(y1 - y2);
    // h = 0;
    // return 0;
}

void Astar_search(const string input_file, int &step_nums, string &way)
{
    ifstream file(input_file);
    if (!file.is_open())
    {
        cout << "Error opening file!" << endl;
        return;
    }

    string line;
    getline(file, line); // 读取第一行
    stringstream ss(line);
    string word;
    vector<string> words;
    while (ss >> word)
    {
        words.push_back(word);
    }
    int M = stoi(words[0]); // 行数
    int N = stoi(words[1]); // 列数
    int T = stoi(words[2]); // T天份额，代表初始能走几步
    // cout << M << " " << N << " " << T << endl;
    pair<int, int> start_point; // 起点
    pair<int, int> end_point;   // 终点
    Map_Cell **Map = new Map_Cell *[M];
    // 加载地图
    for (int i = 0; i < M; i++)
    {
        Map[i] = new Map_Cell[N];
        getline(file, line);
        stringstream ss(line);
        string word;
        vector<string> words;
        while (ss >> word)
        {
            words.push_back(word);
        }
        for (int j = 0; j < N; j++)
        {
            Map[i][j].type = stoi(words[j]);
            if (Map[i][j].type == 3)
            {
                start_point = {i, j};
            }
            else if (Map[i][j].type == 4)
            {
                end_point = {i, j};
            }
        }
    }
    // 以上为预处理部分
    // ------------------------------------------------------------------

    priority_queue<Search_Cell *, vector<Search_Cell *>, CompareF> open_list;
    vector<Search_Cell *> close_list;

    vector<pair<int, int>> direction_lists;
    Search_Cell *start = new Search_Cell(start_point.first, start_point.second, 0, Heuristic_Funtion(start_point.first, start_point.second, end_point.first, end_point.second), T);
    open_list.push(start);
    // search_cell->g = 0;
    // search_cell->h = 0; // Heuristic_Funtion();
    // open_list.push(search_cell);
    map<tuple<int, int, int>, int> cost_map;
    cost_map[{start_point.first, start_point.second, T}] = 0;

    // end_point location
    int end_x = end_point.first;
    int end_y = end_point.second;

    while (!open_list.empty())
    {
        // TODO: A*搜索过程实现
        auto *current_node = open_list.top();
        open_list.pop();
        int cur_x = current_node->x;
        int cur_y = current_node->y;
        // cout << cur_x << " " << cur_y << " " << current_node->supply << endl;
        // find a solution
        if (cur_x == end_x && cur_y == end_y)
        {
            // 打印路径way
            int pre_x = current_node->x;
            int pre_y = current_node->y;
            current_node = current_node->parent;
            if (current_node != nullptr)
                step_nums++;
            while (current_node != nullptr)
            {
                step_nums++;
                if (current_node->x - pre_x == 1)
                    direction_lists.push_back({-1, 0});
                else if (current_node->x - pre_x == -1)
                    direction_lists.push_back({1, 0});
                else if (current_node->y - pre_y == 1)
                    direction_lists.push_back({0, -1});
                else if (current_node->y - pre_y == -1)
                    direction_lists.push_back({0, 1});
                pre_x = current_node->x;
                pre_y = current_node->y;
                current_node = current_node->parent;
            }
            break;
        }

        if (current_node->supply <= 0)
            continue;
        // if not find a solution
        // add new state to the open_list if legal
        vector<pair<int, int>> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (auto dir : directions)
        {
            int next_x = cur_x + dir.first;
            int next_y = cur_y + dir.second;
            if (next_x >= 0 and next_x < M and next_y >= 0 and next_y < N and Map[next_x][next_y].type != 1)
            {
                int new_g = current_node->g + 1;
                int new_supply = current_node->supply - 1;
                if (Map[next_x][next_y].type == 2)
                    new_supply = T;

                tuple<int, int, int> next_state = {next_x, next_y, new_supply};
                if (cost_map.find(next_state) == cost_map.end() || new_g < cost_map[next_state])
                {
                    cost_map[next_state] = new_g;
                    int new_h = Heuristic_Funtion(next_x, next_y, end_x, end_y);
                    Search_Cell *next_node = new Search_Cell(next_x, next_y, new_g, new_h, new_supply, current_node);
                    open_list.push(next_node);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // TODO: 填充step_nums与way
    // step_nums = -1;
    // way = "";
    // cout << direction_lists.size() << endl;
    for (auto it = direction_lists.rbegin(); it != direction_lists.rend(); it++)
    {
        if (it->first == 0 and it->second == 1)
            way = way + "R";
        else if (it->first == 0 and it->second == -1)
            way = way + "L";
        else if (it->first == 1 and it->second == 0)
            way = way + "D";
        else
            way = way + "U";
    }
    // ------------------------------------------------------------------
    // 释放动态内存
    for (int i = 0; i < M; i++)
    {
        delete[] Map[i];
    }
    delete[] Map;
    while (!open_list.empty())
    {
        auto temp = open_list.top();
        delete[] temp;
        open_list.pop();
    }
    for (int i = 0; i < close_list.size(); i++)
    {
        delete[] close_list[i];
    }

    return;
}

void output(const string output_file, int &step_nums, string &way)
{
    ofstream file(output_file);
    if (file.is_open())
    {
        file << step_nums << endl;
        if (step_nums >= 0)
        {
            file << way << endl;
        }

        file.close();
    }
    else
    {
        cerr << "Can not open file: " << output_file << endl;
    }
    return;
}

int main(int argc, char *argv[])
{
    string input_base = "../input/input_";
    string output_base = "../output/output_";
    // input_0为讲义样例，此处不做测试
    for (int i = 1; i < 11; i++)
    {
        int step_nums = -1;
        string way = "";
        auto start = chrono::high_resolution_clock::now();
        Astar_search(input_base + to_string(i) + ".txt", step_nums, way);
        // cout << step_nums << endl;
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end - start;
        cout << "Astar algorithm execution time on the case " << i << " is: " << elapsed.count() << " ms\n";
        output(output_base + to_string(i) + ".txt", step_nums, way);
    }
    return 0;
}