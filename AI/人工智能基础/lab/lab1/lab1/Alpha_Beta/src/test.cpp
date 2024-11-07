#include <fstream>
#include "node.h"
#include <string>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <memory>

using namespace ChineseChess;

int board_cnt = 0;
// 博弈树搜索，depth为搜索深度
int alphaBeta(GameTreeNode &node, int alpha, int beta, int depth, bool isMaximizer)
{
    if (depth == 0 || node.isTerminate())
    {
        return node.getEvaluationScore();
    }
    // TODO: alpha-beta剪枝过程
    std::vector<Move> move_list = node.getBoardClass().getMoves(isMaximizer);
    std::vector<std::vector<char>> cur_board = node.getBoardClass().getBoard();
    
    if(move_list.empty())
    {
        return node.getEvaluationScore();
    }
    Move cur_move = move_list[0];
    if (isMaximizer)
    {
        int max_eval =std::numeric_limits<int>::min();
        for(const Move &step : move_list)
        {
            board_cnt++;
            std::unique_ptr<GameTreeNode> child_node(node.updateBoard(cur_board, step, isMaximizer));
            int score = alphaBeta(*child_node, alpha, beta, depth - 1, false);
            max_eval = std::max(max_eval, score);
            if(score > alpha)
            {
                alpha = score;
                cur_move = step;
            }
            alpha = std::max(alpha, score);
            // if(beta <= alpha)
            //     break;  // 剪枝
        }
        node.best_move = cur_move;
        return max_eval;
    }
    else
    {
        int min_eval = std::numeric_limits<int>::max();
        for(const Move& step : move_list)
        {
            std::unique_ptr<GameTreeNode> child_node(node.updateBoard(cur_board, step, isMaximizer));
            int score = alphaBeta(*child_node, alpha, beta, depth - 1, true);
            min_eval = std::min(min_eval, score);
            if(score < beta)
            {
                beta = score;
                cur_move = step;
            }
            beta = std::min(beta, score);
            // if(alpha >= beta)
            //     break;
        }
        node.best_move = cur_move;
        return min_eval;
    }
}

int test(int index)
{
    board_cnt = 0;
    std::string file_base = "../";
    std::string input_file = file_base + "input/" + std::to_string(index) + ".txt";
    std::string output_file = file_base + "output/" + std::to_string(index) + ".txt";
    std::ifstream file(input_file);
    std::vector<std::vector<char>> board;

    std::string line;
    int n = 0;
    while (std::getline(file, line))
    {
        std::vector<char> row;

        for (char ch : line)
        {
            row.push_back(ch);
        }
        board.push_back(row);
        n++;
        if (n >= 10)
            break;
    }
    file.close();
    // std::cout << "enter here1!" << std::endl;
    GameTreeNode root(true, board, std::numeric_limits<int>::min());

    // std::cout << "finish create and start alphabeta" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    alphaBeta(root, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), 3, true);
    auto end = std::chrono::high_resolution_clock::now();
    Move result = root.best_move;
    std::cout << "score is " << root.best_move.score << std::endl;
    // std::cout << "finish alphabeta " << std::endl;

    // 代码测试
    ChessBoard _board = root.getBoardClass();
    std::vector<std::vector<char>> cur_board = _board.getBoard();

    // for (int i = 0; i < cur_board.size(); i++)
    // {
    //     for (int j = 0; j < cur_board[0].size(); j++)
    //     {
    //         std::cout << cur_board[i][j];
    //     }
    //     std::cout << std::endl;
    // }

    std::vector<Move> red_moves = _board.getMoves(true);
    std::vector<Move> black_moves = _board.getMoves(false);
    std::ofstream output(output_file);
    output << cur_board[result.init_x][result.init_y] << "\n";
    output << " (" << result.init_x << "," << result.init_y << ") \n (" << result.next_x << "," << result.next_y << ")";

    std::cout << "testcase " << std::to_string(index) << ":" << std::endl;

    std::ofstream red_file(file_base + "evaluation/red_" + std::to_string(index) + ".txt");
    std::ofstream black_file(file_base + "evaluation/black_" + std::to_string(index) + ".txt");
    for (int i = 0; i < red_moves.size(); i++)
    {
        red_file << "init: " << red_moves[i].init_x << " " << red_moves[i].init_y << std::endl;
        red_file << "next: " << red_moves[i].next_x << " " << red_moves[i].next_y << std::endl;
        red_file << "score " << red_moves[i].score << std::endl;
    }
    // for (int i = 0; i < red_moves.size(); i++)
    // {
    //     std::cout << "init: " << red_moves[i].init_x << " " << red_moves[i].init_y << std::endl;
    //     std::cout << "next: " << red_moves[i].next_x << " " << red_moves[i].next_y << std::endl;
    //     std::cout << "score " << red_moves[i].score << std::endl;
    // }
    red_file.close(); // 关闭文件
    for (int i = 0; i < black_moves.size(); i++)
    {
        black_file << "init: " << black_moves[i].init_x << " " << black_moves[i].init_y << std::endl;
        black_file << "next: " << black_moves[i].next_x << " " << black_moves[i].next_y << std::endl;
        black_file << "score " << black_moves[i].score << std::endl;
    }
    // for (int i = 0; i < black_moves.size(); i++)
    // {
    //     std::cout << "init: " << black_moves[i].init_x << " " << black_moves[i].init_y << std::endl;
    //     std::cout << "next: " << black_moves[i].next_x << " " << black_moves[i].next_y << std::endl;
    //     std::cout << "score " << black_moves[i].score << std::endl;
    // }
    black_file.close(); // 关闭文件
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

int main()
{
    int time = 0;
    for (int i = 1; i < 11; i++)
    {
        int test_time = test(i);
        std::cout << test_time << " ms"<< std::endl;
        time += test_time;
        std::cout << board_cnt << std::endl;
    }
    std::cout << "the total executing time for all the test cases: " << time << "ms" << std::endl;
    return 0;
}