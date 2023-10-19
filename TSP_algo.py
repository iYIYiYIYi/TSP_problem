import enum
import math
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class AntState(enum.Enum):
    Finished = 0
    Traversing = 1
    Stuck = 2


class AntAlgo:
    def __init__(self, ants_num: int, graph_table, epoch: int, alpha, beta, ant_total_hormone=1.0, evaporate_rate=0.1):
        self.ants_num = ants_num
        self.graph = nx.Graph()

        for i in range(len(graph_table)):
            self.graph.add_node(i)

        for i in range(len(graph_table)):
            for j in range(len(graph_table)):
                if graph_table[i][j] != 0:
                    self.graph.add_edge(i, j, weight=graph_table[i][j], hormone=0)  # weight 权重，hormone 信息素

        self.epoch = epoch
        node_sum = len(self.graph.nodes())
        self.tabu = np.zeros([ants_num, node_sum], dtype=int)  # 蚂蚁已经到访过的表
        self.tabu_edges = []
        for i in range(self.ants_num):
            self.tabu_edges.append([])

        # 随机初始化蚂蚁位置
        self.current_ant_loc = np.random.randint(node_sum, size=ants_num)
        self.start_ant_loc = self.current_ant_loc.copy()
        for loc_i in range(len(self.current_ant_loc)):
            loc = self.current_ant_loc[loc_i]
            self.tabu[loc_i][loc] = -1

        # 蚂蚁走过路径所付出的代价
        self.ant_path_cost = np.zeros(ants_num, dtype=int)
        self.ants_hormone_to_edge = np.zeros(self.ants_num)

        self.alpha = alpha
        self.beta = beta
        self.ant_total_hormone = ant_total_hormone
        self.evaporate_rate = evaporate_rate

        self.thread_pool = ThreadPoolExecutor(max_workers=24)
        self.tasks = []
        self._epoch = 0
        self.graph_pos = nx.spring_layout(self.graph)

    def do_loop(self, fn_draw_plt=None, **kwargs):
        while self._epoch < self.epoch:
            self._epoch += 1
            result = False
            while not result:
                result = self.population_move()

            # 查找最优解
            best_i = 0
            best_v = np.iinfo(np.int32).max
            for index, value in enumerate(self.ant_path_cost):
                if value < best_v and sum(self.tabu[index]) == len(self.graph):
                    best_i = index
                    best_v = value

            if fn_draw_plt is not None:
                fn_draw_plt(self, kwargs)

            # 输出最优路径
            best_path = self.tabu_edges[best_i]
            print("Epoch[%d/%d]: Best path cost: %d " % (self._epoch, self.epoch, best_v))
            print("           path:" + str(best_path))

            # 所有蚂蚁都走完了路径, 清空路径表
            self.tabu = np.zeros_like(self.tabu)
            # self.current_ant_loc = self.start_ant_loc.copy() # 是否需要回到原点
            self.tasks.clear()
            for i in range(self.ants_num):
                loc = self.current_ant_loc[i]
                self.tabu[i][loc] = -1
                self.tabu_edges[i].clear()

    def population_move(self):
        all_finished = 0

        # 函数闭包
        def ant_move_parallel(ant_id: int):
            is_finished = self.ant_move(ant_id)
            if is_finished[0] == AntState.Finished or is_finished[0] == AntState.Stuck:
                total_path_weight = 0
                for e in self.tabu_edges[ant_id]:
                    total_path_weight += self.graph[e[0]][e[1]]['weight']

                self.ant_path_cost[ant_id] = total_path_weight

                h_t_e = self.ant_total_hormone / total_path_weight
                self.ants_hormone_to_edge[ant_id] = h_t_e

            return is_finished

        for ant_id in range(self.ants_num):
            t = self.thread_pool.submit(ant_move_parallel, ant_id=ant_id)
            self.tasks.append(t)

        # 等待任务结束
        for task in self.tasks:
            result = task.result()
            if result[0] == AntState.Finished or result[0] == AntState.Stuck:
                print('Ant['+str(result[1])+'] '+str(result[0]))
                all_finished += 1

        # 蒸发
        for u, v in self.graph.edges():
            self.graph[u][v]['hormone'] = (1 - self.evaporate_rate) * self.graph[u][v]['hormone']

        # 为蚂蚁走过的路添加信息素
        for ant_id in range(self.ants_num):
            if sum(self.tabu[ant_id]) < len(self.graph):
                continue
            hormone_to_edge = self.ants_hormone_to_edge[ant_id]
            for edge in self.tabu_edges[ant_id]:
                self.graph[edge[0]][edge[1]]['hormone'] = hormone_to_edge + self.graph[edge[0]][edge[1]]['hormone']

        if all_finished == self.ants_num:
            return True
        else:
            return False

    # 蚂蚁移动函数
    def ant_move(self, ant_id):
        current_loc = self.current_ant_loc[ant_id]
        cities_nearby = self.graph[current_loc]
        next_city = -1
        gambling_list = []
        for next_potential_city in cities_nearby:
            if self.tabu[ant_id][next_potential_city] == 0:
                # 没访问过, 加入轮盘赌列表
                gambling_list.append((next_potential_city, cities_nearby[next_potential_city]))

        if len(gambling_list) == 0:
            if self.tabu[ant_id][self.start_ant_loc[ant_id]] == -1:
                # 返回初始城市
                next_city = self.start_ant_loc[ant_id]
                if next_city not in cities_nearby.keys():
                    return AntState.Stuck, ant_id
                gambling_list.append((next_city, cities_nearby[next_city]))
            else:
                return AntState.Finished, ant_id

        # 开始轮盘赌
        # 先计算各部分概率以及累计概率
        probabilities = []
        t_all, n_all = 0, 0
        for item in gambling_list:
            t_all += item[1]['weight']
            n_all += item[1]['hormone']

        for i in range(len(gambling_list)):
            t, n = 1 / gambling_list[i][1]['weight'], gambling_list[i][1]['hormone']
            probability = (1 + math.pow(t, self.alpha) * math.pow(n, self.beta)) / (len(gambling_list) +
                    math.pow(t_all, self.alpha) * math.pow(n_all, self.beta))
            probabilities.append(probability)

        t_p = sum(probabilities)
        rate = random.random() * t_p
        pivot = 0
        for i in range(len(gambling_list)):
            pivot += probabilities[i]
            if rate < pivot:
                next_city = gambling_list[i][0]
                break

        self.tabu_edges[ant_id].append((current_loc, next_city))
        self.tabu[ant_id][next_city] = 1
        self.current_ant_loc[ant_id] = next_city
        return AntState.Traversing, ant_id


def fn_draw_plt(obj: AntAlgo, args):
    edge_width = [obj.graph[u][v]['hormone'] * 5 for u, v in obj.graph.edges()]
    nx.draw(obj.graph, width=edge_width, with_labels=True, pos=obj.graph_pos)
    plt.text(-0.85, -0.85, 'Epoch: %d/%d' % (obj._epoch, obj.epoch))
    plt.show()


if __name__ == '__main__':
    city = [
        [0, 5, 1, 1, 1],
        [5, 0, 1, 1, 3],
        [1, 1, 0, 1, 2],
        [1, 1, 1, 0, 12],
        [1, 3, 2, 12, 0],
    ]
    epoch = 10
    ants = AntAlgo(5, city, epoch, 1, 1, 5, 0.5)
    ants.do_loop(fn_draw_plt)
