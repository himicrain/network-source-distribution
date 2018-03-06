
import copy
import sys
import random



class node:
    def __init__(self,id,name):
        self.id = id
        self.name = name


class graph:
    def __init__(self,path,type):
        self.edges = []
        self.nodes = []
        self.node_num = 0
        self.edge_num = 0
        self.edges_info = []
        self.type = type
        self.INFINITY = 1024
        self.get_shortest = None

        topology_datas = open(path)
        self.index_nodes = []
        for line in topology_datas:
            ds = line.strip().split()
            if ds[0] not in self.index_nodes:
                self.index_nodes.append(ds[0])
            if ds[1] not in self.index_nodes:
                self.index_nodes.append(ds[1])
            self.edges_info.append({'start_node':ds[0],'end_node':ds[1],'delay':int(ds[2]),'capacity':int(ds[3]), 'used':0 })

        for i,n in enumerate(self.index_nodes):
            self.nodes.append(node(i,n))

        for i,e in enumerate(self.edges_info):
            n1 = e['start_node']
            n2 = e['end_node']
            i1 = self.index_nodes.index(n1)
            i2 = self.index_nodes.index(n2)
            self.edges_info[i]['start_node'] = node(i1,n1)
            self.edges_info[i]['end_node'] = node(i2,n2)

        self.node_num = len(self.nodes)
        self.edge_num = len(self.edges_info)

        self.init_cost()


        if type == "SHP":
            self.get_graph_matrix_of_shp()
            self.get_shortest = self.shp_sdp
        elif type == "SDP":
            self.get_graph_matrix_of_sdp()
            self.get_shortest = self.shp_sdp
        elif type == "LLP":
            self.get_shortest = self.llp


    def get_graph_matrix_of_shp(self):

        for e in self.edges_info:
            start = e['start_node']
            end = e['end_node']
            cost = 1
            self.edges[start.id][end.id] = cost
            self.edges[end.id][start.id] = cost


    def get_graph_matrix_of_sdp(self):

        for e in self.edges_info:
            start = e['start_node']
            end = e['end_node']
            self.edges[start.id][end.id] = e['delay']
            self.edges[end.id][start.id] = e['delay']

    def init_cost(self):
        for n in range(self.node_num):
            self.edges.append([])
            for nn in range(self.node_num):
                if n == nn:
                    self.edges[n].append(0)
                else:
                    self.edges[n].append(self.INFINITY)

    def init_graph_matrix(self,start_node,end_node):
        ns = self.index_nodes.index(start_node)
        self.affrims = [False for x in self.index_nodes]
        self.affrims[ns] = True
        self.shortests = copy.deepcopy(self.edges[ns])

        self.results_path_tree = []
        for v in self.nodes:
            if self.edges[ns][v.id] == self.INFINITY:
                self.results_path_tree.append(-1)
            else:
                self.results_path_tree.append(ns)

    def get_graph_matrix_of_llp(self):

        for e in self.edges_info:
            start = e['start_node']
            end = e['end_node']
            cost = round(e['used']/e['capacity'],6)
            self.edges[start.id][end.id] = cost
            self.edges[end.id][start.id] = cost

    def llp(self,start_node,end_node):

        self.get_graph_matrix_of_llp()

        ns = self.index_nodes.index(start_node)
        ne = self.index_nodes.index(end_node)
        self.init_graph_matrix(start_node,end_node)

        for i in range(self.node_num - 1):
            min_dist = self.INFINITY
            min_dist_node = 0
            for j in range(len(self.nodes)):
                if self.affrims[j] == False and self.shortests[j] < min_dist:
                    min_dist_node = j
                    min_dist = self.shortests[j]

            self.affrims[min_dist_node] = True

            for j in range(self.node_num):
                # 唯一不同是，上面那个是通过比较大小更新，这个是累加之后更新
                max_one = max(self.shortests[min_dist_node] , self.edges[min_dist_node][j])
                if self.affrims[j] == False and max(self.shortests[min_dist_node] , self.edges[min_dist_node][j]) < self.shortests[j]:
                    self.shortests[j] =  max_one
                    self.results_path_tree[j] = min_dist_node

        return self.tree_to_path(ns, ne)


    def shp_sdp(self,start_node,end_node):
        ns = self.index_nodes.index(start_node)
        ne = self.index_nodes.index(end_node)

        self.init_graph_matrix(start_node, end_node)

        for i in range(self.node_num - 1):
            min_dist = self.INFINITY
            min_dist_node = 0
            for j in range(len(self.nodes)):
                if self.affrims[j] == False and self.shortests[j] < min_dist:
                    min_dist_node = j
                    min_dist = self.shortests[j]

            self.affrims[min_dist_node] = True

            for j in range(self.node_num):
                # 唯一不同是，上面那个是通过比较大小更新，这个是累加之后更新
                temp = self.shortests[min_dist_node] + self.edges[min_dist_node][j]
                if self.affrims[j] == False and temp < self.shortests[j]:
                    self.shortests[j] = temp
                    self.results_path_tree[j] = min_dist_node
        return self.tree_to_path( ns, ne)

    def tree_to_path(self,ns,ne):
        parse_path = []
        parse_path.append(ne)
        for i in range(len(self.results_path_tree)):
            ne = self.results_path_tree[ne]
            parse_path.append(ne)
            if ne == ns:
                break

        for k,v in enumerate(parse_path):
            parse_path[k] = self.index_nodes[v]


        i = 0
        j = len(parse_path)-1

        while i <= j:
            temp = parse_path[i]
            parse_path[i] = parse_path[j]
            parse_path[j] = temp
            i+=1
            j-=1

        return parse_path


class WorkLoad:
    def __init__(self,workload,rate_num):
        workload_all = open(workload)
        self.Works = []
        self.rate_num = rate_num
        self.noused = 0

        for work in workload_all:
            datas = work.strip().split()

            '''
            if float(datas[3]) < 1/self.rate_num:
                #self.noused += 1
                #continue
                pass
            '''
            start_node = datas[1]
            end_node = datas[2]
            start_time = round(float(datas[0]), 6)
            duration = round(float(datas[3]), 6)
            #finish_time = round(int(float(datas[3]) * rate_num) * round(1 / rate_num, 6) + float(datas[0]), 6)
            finish_time = round(float(datas[3]) + float(datas[0]) + 1/self.rate_num,6)
            is_blocked = 0
            current_exec_time = round(float(datas[0]), 6)
            shortest_path = []
            temp = {'start_node':start_node,'end_node':end_node,'duration':duration,'start_time':start_time,'finish_time':finish_time,'is_blocked':is_blocked,'exec_time':current_exec_time,'shortest':shortest_path}
            self.Works.append(temp)

        self.the_work_of_handling = []
        self.the_work_of_waiting = copy.deepcopy(self.Works)



    def check_will_join(self,time):

        works = []
        count = 0
        for work in self.the_work_of_waiting:
            if work['start_time'] <= time:
                works.append(work)
                count += 1
            else:
                break
        for c in range(count):
            self.the_work_of_waiting.pop(0)

        if works != []:
            self.the_work_of_handling.extend(works)
            self.the_work_of_handling.sort(key=lambda x : x['exec_time'])

        return works

    def check_will_finish(self,time):
        works = []
        for work in self.the_work_of_handling:
            if work['exec_time'] > work['finish_time']:
                works.append(work)
        for work in works:
            self.the_work_of_handling.remove(work)
        return works



class Driver:
    def __init__(self, net_scheme, routing_scheme, topo_file, work_file, rate_num,work_load,graph_obj):
            self.Rate_num = int(rate_num)
            self.RATE = 1 / self.Rate_num

            self.netwok_scheme = net_scheme
            self.routing_scheme = routing_scheme

            self.workload_file = work_file
            self.topology_file = topo_file

            self.work_load = work_load
            self.graph_obj = graph_obj

            # 总的delay
            self.VD = 0
            # 总的hops
            self.VH = 0
            # 总的packets
            self.VP = 0
            # 总的circuit
            self.VC = 0
            # 所有blocketed的
            self.VB = 0
            # blocked的circuit
            self.VBC = 0
            # 成功的packet
            self.VPS = 0
            self.VCS = 0
            self.INFINITY = 1024

            self.get_shortest_path = None
            self.main = None

            if self.netwok_scheme == "PACKET":
                self.main = self.main_of_packet
            elif self.netwok_scheme == "CIRCUIT":
                self.main = self.main_of_circuit

    def print_all(self):

        print('total number of virtual circuit requests:            %s' %(self.VC))
        print('total number of packets:                             %s' %(self.VP))
        print('number of successfully routed packets:               %s' %(self.VPS))
        print('percentage of successfully routed packets:           %s' %(round(float(self.VPS) * 100 / self.VP, 6)))
        print('number of blocked packets:                           %s' %(self.VB))
        print('percentage of blocked packets:                       %s' %(round(float(self.VB) * 100 / self.VP, 6)))
        print('average number of hops per circuit:                  %s' %(round(float(self.VH) / self.VC, 2)))
        print('average cumulative propagation delay per circuit:    %s' %(round(float(self.VD) / self.VC, 2)))

    #circuit下整个流程
    def main_of_circuit(self):

        #self.VC = self.work_load.noused

        Time = 0
        # 开始执行任务
        while True:
            #检查是否存在到达的任务
            request_temp = self.work_load.check_will_join(Time)
            #用于暂时存储需要申请资源的任务的路径
            final_request = []
            for k,path in enumerate(request_temp):
                sp = self.graph_obj.get_shortest(path['start_node'],path['end_node'])
                pi = self.work_load.the_work_of_handling.index(request_temp[k])

                self.work_load.the_work_of_handling[pi]['shortest'] = sp
                request_temp[k]['shortest']= sp
                final_request.append(sp)

            #如果请求资源的路径发生了block，那么标记为False ,如果可以请求资源，那么标记True
            flags= []
            for k,path in enumerate(final_request):
                flag = 1
                for i in range(len(path)-1):
                    for e in self.graph_obj.edges_info:
                        if {path[i], path[i + 1]} == {e['start_node'].name, e['end_node'].name} and  e['used'] >= e['capacity']:
                            flag = 0
                            break
                    if flag == 0:
                        flags.append(False)
                        break
                if flag != 0:
                    flags.append(True)
            #针对上一步的标记，记录信息，同时申请资源，和丢弃任务
            for k,f in enumerate(flags):
                e = request_temp[k]
                upadte = int(self.Rate_num*e['duration'])

                #更新
                self.VP += upadte
                self.VC += 1

                #如果blocked了，那么记录，同时把任务丢弃
                if f == False:
                    self.VBC += 1
                    self.VB += upadte
                    self.work_load.the_work_of_handling.remove(e)
                #申请资源
                else:
                    self.VPS += upadte
                    self.VCS += 1
                    temp_shorest_path = e['shortest']

                    #对每条边申请资源
                    for i in range(len(temp_shorest_path)-1):
                        for ef in self.graph_obj.edges_info:
                            if {temp_shorest_path[i], temp_shorest_path[i + 1]} == {ef['start_node'].name,ef['end_node'].name}:
                                ef['used'] += 1
                                self.VD += ef['delay']
                        self.VH += 1

            #更新每个任务的状态信息
            for work in self.work_load.the_work_of_handling:
                work['exec_time'] = round(work['exec_time']+self.RATE, 6)

            # 更新当前时间
            Time = round(Time + self.RATE, 6)

            # 检查是否存在结束的任务
            finish_temp = self.work_load.check_will_finish(Time)
            for w in finish_temp:
                #对所有要完成的任务进行资源释放
                temp_shorest_path = w['shortest']
                for i in range(len(temp_shorest_path) - 1):
                    for ef in self.graph_obj.edges_info:
                        if {temp_shorest_path[i], temp_shorest_path[i + 1]} == {ef['start_node'].name,ef['end_node'].name}:
                            ef['used'] -= 1

            # 判断所有任务完成
            if self.work_load.the_work_of_handling == [] and self.work_load.the_work_of_waiting == []:
                break

            #可以注释掉
            print('current time :  ',Time)

        #打印统计信息
        self.print_all()


    #packet模式下，处理流程
    def main_of_packet(self):
            Time = 0
            #self.VC -= self.work_load.noused
            #self.VP -= self.work_load.noused

            while True:
                #检查是否存在开始的任务
                self.work_load.check_will_join(Time)
                #获取需要处理的任务
                request_temp = self.work_load.the_work_of_handling
                #对每个任务进行处理，申请和释放资源
                for k,v in enumerate(request_temp):
                    #如果不是刚开始 或者上一次发生了blocked的任务，那么首先进行资源释放
                    if v['exec_time'] != v['start_time'] and v['is_blocked']!= 1:
                        temp_shortest = v['shortest']
                        for i in range(len(temp_shortest)-1):
                            for ef in self.graph_obj.edges_info:
                                if {temp_shortest[i],temp_shortest[i+1]} == {ef['start_node'].name, ef['end_node'].name}:
                                    ef['used'] -= 1

                    self.VP += 1
                    self.VC += 1
                    #获取最短路径
                    shortest_path = self.graph_obj.get_shortest(v['start_node'],v['end_node'])
                    v['shortest'] = shortest_path

                    #检查是否可以请求资源
                    blocked = 0
                    for i in range(len(shortest_path) - 1):
                        for ef in self.graph_obj.edges_info:
                            if {shortest_path[i],shortest_path[i + 1]} == {ef['start_node'].name, ef['end_node'].name} :
                                if ef['used'] >= ef['capacity']:
                                    blocked = 1
                                    break

                    #如果block了，那么记录
                    if blocked == 1:
                        self.VB += 1
                        v['is_blocked'] = 1

                    #如果没有block 那么l[6]=1 就是任务没有被block的标志
                    else:
                        v['is_blocked'] = 0
                        self.VPS += 1
                        self.VCS += 1
                        #为该路径请求资源
                        #self.sources.acquireLink(path)
                        for i in range(len(shortest_path) - 1):

                            for ef in self.graph_obj.edges_info:
                                if {shortest_path[i],shortest_path[i+1]} == {ef['start_node'].name, ef['end_node'].name}:
                                    ef['used'] += 1
                                    self.VD +=ef['delay']
                            self.VH += 1

                # 对每个任务已经执行的时间进行更新
                for work in self.work_load.the_work_of_handling:
                    work['exec_time'] = round(work['exec_time']+self.RATE,6)


                # 更新当前时间
                Time = round(Time + self.RATE, 6)

                # 存在即将结束的任务
                finish_temp = self.work_load.check_will_finish(Time)
                for w in finish_temp:
                    #如果该请求上一次被block了，那不进行释放
                    if w['is_blocked'] == 1:
                        continue
                    #释放完成的任务的资源
                    temp_shorest_path = w['shortest']
                    for i in range(len(temp_shorest_path) - 1):
                        for ef in self.graph_obj.edges_info:
                            if {temp_shorest_path[i], temp_shorest_path[i + 1]} == {ef['start_node'].name, ef['end_node'].name}:
                                ef['used'] -= 1

                if self.work_load.the_work_of_handling == [] and self.work_load.the_work_of_waiting == []:
                    break

                #这句话可以注释掉，只是为了方便观察运行到哪了
                print('current time :  ', Time)

            # 打印统计信息
            self.print_all()



if __name__ == "__main__":
    import time
    start = time.time()

    net_scheme= sys.argv[1]
    routing_scheme= sys.argv[2]
    topo_file = sys.argv[3]
    work_file= sys.argv[4]
    rate_num = sys.argv[5]
    #用于存储所有的任务
    work_load = WorkLoad(work_file,int(rate_num))
    #存储图信息
    graph_obj = graph(topo_file,routing_scheme)
    #驱动程序
    driver_obj = Driver(net_scheme, routing_scheme, topo_file, work_file, rate_num,work_load,graph_obj)
    driver_obj.main()

    print(time.time()-start)

