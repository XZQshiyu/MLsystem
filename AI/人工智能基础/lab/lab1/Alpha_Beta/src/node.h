#include <vector>
#include <map>
#include <limits>
#include <iostream>
#include <string>

namespace ChineseChess
{
    // 棋力评估，这里的棋盘方向和输入棋盘方向不同，在使用时需要仔细
    // 生成合法动作代码部分已经使用，经过测试是正确的，大家可以参考
    std::vector<std::vector<int>> JiangPosition = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {5, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    std::vector<std::vector<int>> ShiPosition = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    std::vector<std::vector<int>> XiangPosition = {
        {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 3, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
    };

    std::vector<std::vector<int>> MaPosition = {
        {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
        {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
        {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
        {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
        {2, -10, 4, 10, 15, 16, 12, 11, 6, 2},
        {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
        {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
        {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
        {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
    };

    std::vector<std::vector<int>> PaoPosition = {
        {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
        {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
        {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
        {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
        {3, 2, 5, 0, 4, 4, 4, -4, -7, -6},
        {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
        {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
        {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
    };

    std::vector<std::vector<int>> JuPosition = {
        {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
        {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
        {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
        {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
        {0, 0, 12, 14, 15, 15, 16, 16, 33, 14},
        {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
        {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
        {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
        {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
    };

    std::vector<std::vector<int>> BingPosition = {
        {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
        {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
        {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
        {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
        {0, 0, 0, 6, 7, 40, 42, 55, 70, 4},
        {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
        {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
        {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
        {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
    };

    // 棋子价值评估
    std::map<std::string, int> piece_values = {
        {"Jiang", 10000},
        {"Shi", 10},
        {"Xiang", 30},
        {"Ma", 300},
        {"Ju", 500},
        {"Pao", 300},
        {"Bing", 90}};

    // 行期可能性评估，这里更多是对下一步动作的评估
    std::map<std::string, int> next_move_values = {
        {"Jiang", 9999},
        {"Ma", 100},
        {"Ju", 500},
        {"Pao", 100},
        {"Bing", -20}};

    // 动作结构体，每个动作设置score，可以方便剪枝
    struct Move
    {
        int init_x;
        int init_y;
        int next_x;
        int next_y;
        int score;
    };

    // 定义棋盘上的棋子结构体
    struct ChessPiece
    {
        char name;          // 棋子名称
        int init_x, init_y; // 棋子的坐标
        bool color;         // 棋子阵营 true为红色、false为黑色
    };

    // 定义棋盘类
    class ChessBoard
    {
    private:
        int sizeX, sizeY;                     // 棋盘大小，固定
        std::vector<ChessPiece> pieces;       // 棋盘上所有棋子
        std::vector<std::vector<char>> board; // 当前棋盘、二维数组表示
        std::vector<Move> red_moves;          // 红方棋子的合法动作
        std::vector<Move> black_moves;        // 黑方棋子的合法动作
    public:
        // 判断是否是敌方棋子
        bool is_enemy(char piece, bool color)
        {
            return ((isupper(piece) and !color) or (islower(piece) and color));
        }
        // 初始化棋盘，提取棋盘上棋子，并生成所有合法动作
        void initializeBoard(const std::vector<std::vector<char>> &init_board)
        {
            board = init_board;
            sizeX = board.size();
            sizeY = board[0].size();
            // std::cout << sizeX << " " << sizeY << std::endl;
            // for (int i = 0; i < sizeX; i++)
            // {
            //     for (int j = 0; j < sizeY; j++)
            //         std::cout << board[i][j];
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;
            for (int i = 0; i < sizeX; ++i)
            {
                for (int j = 0; j < sizeY; ++j)
                {
                    char pieceChar = board[i][j];
                    if (pieceChar == '.')
                        continue;
                    // std::cout << i << " " << j << " " << pieceChar << std::endl;
                    ChessPiece piece;
                    piece.init_x = i;
                    piece.init_y = j;
                    piece.color = (pieceChar >= 'A' && pieceChar <= 'Z');
                    piece.name = pieceChar;
                    pieces.push_back(piece);

                    switch (pieceChar)
                    {
                    case 'R':
                        generateJuMoves(i, j, piece.color);
                        break;
                    case 'C':
                        generatePaoMoves(i, j, piece.color);
                        break;
                    case 'N':
                        generateMaMoves(i, j, piece.color);
                        break;
                    case 'B':
                        generateXiangMoves(i, j, piece.color);
                        break;
                    case 'A':
                        generateShiMoves(i, j, piece.color);
                        break;
                    case 'K':
                        generateJiangMoves(i, j, piece.color);
                        break;
                    case 'P':
                        generateBingMoves(i, j, piece.color);
                        break;
                    case 'r':
                        generateJuMoves(i, j, piece.color);
                        break;
                    case 'c':
                        generatePaoMoves(i, j, piece.color);
                        break;
                    case 'n':
                        generateMaMoves(i, j, piece.color);
                        break;
                    case 'b':
                        generateXiangMoves(i, j, piece.color);
                        break;
                    case 'a':
                        generateShiMoves(i, j, piece.color);
                        break;
                    case 'k':
                        generateJiangMoves(i, j, piece.color);
                        break;
                    case 'p':
                        generateBingMoves(i, j, piece.color);
                        break;
                    default:
                        break;
                    }
                }
            }
        }

        // 生成车的合法动作
        void generateJuMoves(int x, int y, bool color)
        {
            // 前后左右分别进行搜索，遇到棋子停止，不同阵营可以吃掉
            std::vector<Move> JuMoves;
            for (int i = x + 1; i < sizeX; i++)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[i][y] != '.')
                {
                    bool cur_color = (board[i][y] >= 'A' && board[i][y] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move);
            }

            for (int i = x - 1; i >= 0; i--)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[i][y] != '.')
                {
                    bool cur_color = (board[i][y] >= 'A' && board[i][y] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move);
            }

            for (int j = y + 1; j < sizeY; j++)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[x][j] != '.')
                {
                    bool cur_color = (board[x][j] >= 'A' && board[x][j] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move);
            }

            for (int j = y - 1; j >= 0; j--)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[x][j] != '.')
                {
                    bool cur_color = (board[x][j] >= 'A' && board[x][j] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                JuMoves.push_back(cur_move);
            }
            // std::cout << "enter here for Ju" << std::endl;
            for (int i = 0; i < JuMoves.size(); i++)
            {
                // std::cout << color << std::endl;
                // std::cout << "pre position: " << JuMoves[i].init_x << " " << JuMoves[i].init_y << std::endl;
                // std::cout << "new position: " << JuMoves[i].next_x << " " << JuMoves[i].next_y << std::endl;
                if (color)
                {
                    JuMoves[i].score = JuPosition[JuMoves[i].next_y][9 - JuMoves[i].next_x] - JuPosition[y][9 - x];
                    red_moves.push_back(JuMoves[i]);
                }
                else
                {
                    JuMoves[i].score = JuPosition[JuMoves[i].next_y][JuMoves[i].next_x] - JuPosition[y][x];
                    black_moves.push_back(JuMoves[i]);
                }
            }
            // std::cout << "finish here "<<std::endl;
        }

        // 生成马的合法动作
        void generateMaMoves(int x, int y, bool color)
        {
            // 便利所有可能动作，筛选
            std::vector<Move> MaMoves;
            int dx[] = {2, 1, -1, -2, -2, -1, 1, 2};
            int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};
            // 简化，不考虑拌马脚
            // TODO: 可以实现拌马脚过程
            for (int i = 0; i < 8; i++)
            {
                Move cur_move;
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx < 0 || nx >= sizeX || ny < 0 || ny >= sizeY)
                    continue;
                // 拌马脚
                if (board[x + dx[i] / 2][y + dy[i] / 2] != '.')
                    continue;
                
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                cur_move.score = 0;
                if (board[nx][ny] != '.')
                {
                    // 注意棋盘坐标系，这里nx、ny相反是正确的
                    bool cur_color = (board[nx][ny] >= 'A' && board[nx][ny] <= 'Z');
                    if (cur_color != color)
                    {
                        MaMoves.push_back(cur_move);
                    }
                    continue;
                }
                MaMoves.push_back(cur_move);
            }
            for (int i = 0; i < MaMoves.size(); i++)
            {
                // std::cout << color << std::endl;
                // std::cout << "pre position: " << MaMoves[i].init_x << " " << MaMoves[i].init_y << std::endl;
                // std::cout << "new position: " << MaMoves[i].next_x << " " << MaMoves[i].next_y << std::endl;
                if (color)
                {
                    MaMoves[i].score = MaPosition[MaMoves[i].next_y][9 - MaMoves[i].next_x] - MaPosition[y][9 - x];
                    red_moves.push_back(MaMoves[i]);
                }
                else
                {
                    MaMoves[i].score = MaPosition[MaMoves[i].next_y][MaMoves[i].next_x] - MaPosition[y][x];
                    black_moves.push_back(MaMoves[i]);
                }
            }
        }

        // 生成炮的合法动作
        void generatePaoMoves(int x, int y, bool color)
        {
            // 和车生成动作相似，需要考虑炮翻山吃子的情况
            std::vector<Move> PaoMoves;
            // TODO:
            // std::cout << "Pao : " << x << " " << y << std::endl;
            int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
            for (int dir = 0; dir < 4; dir++)
            {
                bool Jumped = false;
                for (int step = 1;; step++)
                {
                    int nx = x + directions[dir][0] * step;
                    int ny = y + directions[dir][1] * step;
                    if (nx < 0 or nx >= sizeX or ny < 0 or ny >= sizeY)
                        break;
                    char target = board[nx][ny];
                    if (target == '.')
                    {
                        if (!Jumped)
                        {
                            Move cur_move = {x, y, nx, ny, 0};
                            PaoMoves.push_back(cur_move);
                        }
                    }
                    else
                    {
                        if (!Jumped)
                            Jumped = true;
                        else
                        {
                            bool cur_color = (target >= 'A' && target <= 'Z');
                            if (cur_color != color)
                            {
                                Move cur_move = {x, y, nx, ny, 0};
                                PaoMoves.push_back(cur_move);
                            }
                            break;
                        }
                    }
                }
            }

            for (int i = 0; i < PaoMoves.size(); i++)
            {
                // std::cout << color << std::endl;
                // std::cout << "pre position: " << PaoMoves[i].init_x << " " << PaoMoves[i].init_y << std::endl;
                // std::cout << "new position: " << PaoMoves[i].next_x << " " << PaoMoves[i].next_y << std::endl;
                if (color)
                {
                    PaoMoves[i].score = PaoPosition[PaoMoves[i].next_y][9 - PaoMoves[i].next_x] - PaoPosition[y][9 - x];
                    red_moves.push_back(PaoMoves[i]);
                }
                else
                {
                    PaoMoves[i].score = PaoPosition[PaoMoves[i].next_y][PaoMoves[i].next_x] - PaoPosition[y][x];
                    black_moves.push_back(PaoMoves[i]);
                }
            }
        }

        // 生成相的合法动作
        void generateXiangMoves(int x, int y, bool color)
        {
            // std::cout << "Xiang " << x << " " << y << std::endl;
            std::vector<Move> XiangMoves;
            // TODO:
            int dx[] = {2, 2, -2, -2};
            int dy[] = {2, -2, 2, -2};
            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int ny = y + dy[i];
                // std::cout << nx << " " << ny << std::endl;
                // 过河了
                if((x < 5 and nx >= 5) or (x >= 5 and nx < 5))
                    continue;
                if (nx >= 0 and nx < sizeX and ny >= 0 and ny < sizeY)
                {
                    char target = board[nx][ny];
                    // 检查是否合法，即中间田字格中间是否有棋子
                    if (board[x + dx[i] / 2][y + dy[i] / 2] == '.')
                    {
                        if (target == '.' or is_enemy(target, color))
                        {
                            Move cur_move = {x, y, nx, ny, 0};
                            XiangMoves.push_back(cur_move);
                        }
                    }
                }
            }
            // 更新分数
            // std::cout << "别tm告诉我进入这里了" << std::endl;
            // std::cout << "好好好，这么玩是吧" << std::endl;
            // std::cout << XiangPosition.size() << " " << XiangPosition[0].size() << std::endl;
            for (int i = 0; i < XiangMoves.size(); i++)
            {
                // std::cout << color << std::endl;
                // std::cout << "pre position: " << XiangMoves[i].init_x << " " << XiangMoves[i].init_y << std::endl;
                // std::cout << "new position: " << XiangMoves[i].next_x << " " << XiangMoves[i].next_y << std::endl;
                if (color)
                {
                    XiangMoves[i].score = XiangPosition[XiangMoves[i].next_y][9 - XiangMoves[i].next_x] - XiangPosition[y][9 - x];
                    red_moves.push_back(XiangMoves[i]);
                }
                else
                {
                    XiangMoves[i].score = XiangPosition[XiangMoves[i].next_y][XiangMoves[i].next_x] - XiangPosition[y][x];
                    black_moves.push_back(XiangMoves[i]);
                }
            }
        }

        // 生成士的合法动作
        void generateShiMoves(int x, int y, bool color)
        {
            // std::cout << "shi " << x << " " << y << " " << std::endl;
            std::vector<Move> ShiMoves;
            // TODO:
            int dx[] = {1, 1, -1, -1};
            int dy[] = {1, -1, 1, -1};
            int limitX1 = color ? 7 : 0;
            int limitX2 = color ? 9 : 2;
            int limitY = 3;
            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx >= limitX1 and nx <= limitX2 and ny >= limitY and ny <= limitY + 2)
                {
                    char target = board[nx][ny];
                    if (target == '.' or is_enemy(target, color))
                    {
                        Move cur_move = {x, y, nx, ny, 0};
                        ShiMoves.push_back(cur_move);
                    }
                }
            }

            for (int i = 0; i < ShiMoves.size(); i++)
            {
                // std::cout << color << std::endl;
                // std::cout << "pre position: " << ShiMoves[i].init_x << " " << ShiMoves[i].init_y << std::endl;
                // std::cout << "new position: " << ShiMoves[i].next_x << " " << ShiMoves[i].next_y << std::endl;
                if (color)
                {
                    ShiMoves[i].score = ShiPosition[ShiMoves[i].next_y][9 - ShiMoves[i].next_x] - ShiPosition[y][9 - x];
                    red_moves.push_back(ShiMoves[i]);
                }
                else
                {
                    ShiMoves[i].score = ShiPosition[ShiMoves[i].next_y][ShiMoves[i].next_x] - ShiPosition[y][x];
                    black_moves.push_back(ShiMoves[i]);
                }
            }
        }

        // 生成将的合法动作
        void generateJiangMoves(int x, int y, bool color)
        {
            std::vector<Move> JiangMoves;
            // std::cout << x << " " << y << " " << std::endl;
            // TODO:
            int dx[] = {0, 1, 0, -1};
            int dy[] = {1, 0, -1, 0};
            // 离开9宫格，不合法
            int limitX1 = color ? 7 : 0;
            int limitX2 = color ? 9 : 2;
            int limitY = 3;
            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx >= limitX1 and nx <= limitX2 and ny >= limitY and ny <= limitY + 2)
                {
                    char target = board[nx][ny];
                    if (target == '.' or is_enemy(target, color))
                    {
                        Move cur_move = {x, y, nx, ny, 0};
                        JiangMoves.push_back(cur_move);
                    }
                }
            }
            // std::cout << JiangMoves.size() << std::endl;
            // std::cout << "enter here?" << std::endl;
            for (int i = 0; i < JiangMoves.size(); i++)
            {
                // std::cout << color << std::endl;
                // std::cout << "pre position: " << JiangMoves[i].init_x << " " << JiangMoves[i].init_y << std::endl;
                // std::cout << "new position: " << JiangMoves[i].next_x << " " << JiangMoves[i].next_y << std::endl;
                if (color)
                {
                    JiangMoves[i].score = JiangPosition[JiangMoves[i].next_y][9 - JiangMoves[i].next_x] - JiangPosition[y][9 - x];
                    red_moves.push_back(JiangMoves[i]);
                }
                else
                {
                    JiangMoves[i].score = JiangPosition[JiangMoves[i].next_y][JiangMoves[i].next_x] - JiangPosition[y][x];
                    black_moves.push_back(JiangMoves[i]);
                }
            }
            // std::cout << "leave here?" << std::endl;
        }

        // 生成兵的合法动作
        void generateBingMoves(int x, int y, bool color)
        {
            // std::cout << "Bing: " << x << " " << y << std::endl;
            // 需要分条件考虑，小兵在过楚河汉界之前只能前进，之后可以左右前
            std::vector<Move> BingMoves;
            // TODO:
            int forward = color ? -1 : 1;
            int midline = color ? 5 : 4;
            // 先看是否能forward
            if (x + forward >= 0 and x + forward < sizeX)
            {
                char target = board[x + forward][y];
                if (target == '.' or is_enemy(target, color))
                {
                    Move cur_move = {x, y, x + forward, y, 0};
                    BingMoves.push_back(cur_move);
                }
            }
            // 如果已经过河，尝试左右移动
            if ((color and x < midline) or (!color and x > midline))
            {
                int sideways[] = {-1, 1};
                for (int side : sideways)
                {
                    if (y + side >= 0 and y + side < sizeY)
                    {
                        char target = board[x][y + side];
                        if (target == '.' or is_enemy(target, color))
                        {
                            Move cur_move = {x, y, x, y + side, 0};
                            BingMoves.push_back(cur_move);
                        }
                    }
                }
            }

            for (int i = 0; i < BingMoves.size(); i++)
            {
                // std::cout << color << std::endl;
                // std::cout << "pre position: " << BingMoves[i].init_x << " " << BingMoves[i].init_y << std::endl;
                // std::cout << "new position: " << BingMoves[i].next_x << " " << BingMoves[i].next_y << std::endl;
                if (color)
                {
                    BingMoves[i].score = BingPosition[BingMoves[i].next_y][9 - BingMoves[i].next_x] - BingPosition[y][9 - x];
                    red_moves.push_back(BingMoves[i]);
                }
                else
                {
                    BingMoves[i].score = BingPosition[BingMoves[i].next_y][BingMoves[i].next_x] - BingPosition[y][x];
                    black_moves.push_back(BingMoves[i]);
                }
            }
        }

        // 终止判断
        bool judgeTermination()
        {
            // TODO:
            for (auto &piece : pieces)
            {
                if (piece.name == 'K' and !piece.color)
                    return true;
                if (piece.name == 'k' and piece.color)
                    return true;
            }
            return false;
        }

        // 棋盘分数评估，根据当前棋盘进行棋子价值和棋力评估，max玩家减去min玩家分数
        int evaluateNode()
        {
            // TODO:
            int red_score = 0;
            int black_score = 0;
            for (auto &piece : pieces)
            {
                char piecechar = board[piece.init_x][piece.init_y];
                int x = piece.init_x;
                int y = piece.init_y;
                // 根据类型获取价值
                switch (piecechar)
                {
                case 'k':
                    black_score += JiangPosition[y][x];
                    black_score += piece_values["Jiang"];
                    break;
                case 'a':
                    black_score += ShiPosition[y][x];
                    black_score += piece_values["Shi"];
                    break;
                case 'r':
                    black_score += JuPosition[y][x];
                    black_score += piece_values["Ju"];
                    break;
                case 'c':
                    black_score += PaoPosition[y][x];
                    black_score += piece_values["Pao"];
                    break;
                case 'n':
                    black_score += MaPosition[y][x];
                    black_score += piece_values["Ma"];
                    break;
                case 'b':
                    black_score += XiangPosition[y][x];
                    black_score += piece_values["Xiang"];
                    break;
                case 'p':
                    black_score += BingPosition[y][x];
                    black_score += piece_values["Bing"];
                    break;
                case 'K':
                    red_score += JiangPosition[y][9 - x];
                    red_score += piece_values["Jiang"];
                    break;
                case 'A':
                    red_score += ShiPosition[y][9 - x];
                    red_score += piece_values["Shi"];
                    break;
                case 'R':
                    red_score += JuPosition[y][9 - x];
                    red_score += piece_values["Ju"];
                    break;
                case 'C':
                    red_score += PaoPosition[y][9 - x];
                    red_score += piece_values["Pao"];
                    break;
                case 'N':
                    red_score += MaPosition[y][9 - x];
                    red_score += piece_values["Ma"];
                    break;
                case 'B':
                    red_score += XiangPosition[y][9 - x];
                    red_score += piece_values["Xiang"];
                    break;
                case 'P':
                    red_score += BingPosition[y][9 - x];
                    red_score += piece_values["Bing"];
                    break;
                default :
                    break;
                }
            }
            return red_score - black_score;
        }

        // 测试接口
        std::vector<Move> getMoves(bool color)
        {
            if (color)
                return red_moves;
            return black_moves;
        }

        std::vector<ChessPiece> getChessPiece()
        {
            return pieces;
        }

        std::vector<std::vector<char>> getBoard()
        {
            return board;
        }
    };

    // 定义博弈树节点类
    class GameTreeNode
    {
    private:
        bool color;                           // 当前玩家类型，true为红色方、false为黑色方
        ChessBoard board;                     // 当前棋盘状态
        std::vector<GameTreeNode *> children; // 子节点列表
        int evaluationScore;                  // 棋盘评估分数

    public:
        Move best_move;
        // 构造函数
        GameTreeNode(bool color, std::vector<std::vector<char>> initBoard, int evaluationScore)
            : color(color), evaluationScore(evaluationScore)
        {
            // std::cout << "start initial" << std::endl;
            board.initializeBoard(initBoard);
            // std::cout << "finished initial" << std::endl;
            std::vector<Move> moves = board.getMoves(color);
            children.clear();
            std::vector<std::vector<char>> cur_board = board.getBoard();

            // 为合法动作创建子节点
            // for (int i = 0; i < moves.size(); i++) {
            //     GameTreeNode* child = updateBoard(cur_board, moves[i], color);
            //     children.push_back(child);
            // }
        }

        // 根据当前棋盘和动作构建新棋盘（子节点）
        GameTreeNode *updateBoard(std::vector<std::vector<char>> cur_board, Move move, bool color)
        {
            // TODO:

            // 放置棋子到新位置
            cur_board[move.next_x][move.next_y] = cur_board[move.init_x][move.init_y];
            // 清除初始位置
            cur_board[move.init_x][move.init_y] = '.';
            ChessBoard newChessBoard;
            newChessBoard.initializeBoard(cur_board);
            int evaluation_score = newChessBoard.evaluateNode();
            // std::cout << evaluation_score <<std::endl;
            GameTreeNode *newNode = new GameTreeNode(!color, cur_board, evaluation_score);
            return newNode;
        }

        // 返回节点评估分数
        int getEvaluationScore()
        {
            evaluationScore = board.evaluateNode();
            return evaluationScore;
        }

        // 返回棋盘类
        ChessBoard getBoardClass()
        {
            return board;
        }

        // 辅助函数，判断是否终止
        bool isTerminate()
        {
            return board.judgeTermination();
        }
        ~GameTreeNode()
        {
            for (GameTreeNode *child : children)
            {
                delete child;
            }
        }
    };

}