
import copy
import sys
import random

class Source:
    def __init__(self,path):
        self.E = []
        # 初始化边集合，和 nodes 集合， topology.txt
        file = open(path)
        for line in file:
            lines = line.strip().split()
            self.E.append([lines[0], lines[1], int(lines[2]), int(lines[3]), 0])
        file.close()


    def isBlocked(self,t_e):
        for i in range(len(t_e) - 1):
            flag = 0
            for e in self.E:
                if {e[0],e[1]} == {t_e[i],t_e[i+1]} and e[4] < e[3]:
                    flag = 1
            if flag == 0:
                return 1
        return 0
    def acquireLink(self,requestPath):

        for i in range(len(requestPath) - 1):
            for e in self.E:
                if {requestPath[i], requestPath[i + 1]} == {e[0], e[1]}:
                    e[4] += 1
                    break

    def acquire(self,requestPaths,requestList,Runnings):
        temp = []
        for k, v in enumerate(requestPaths):
            if v ==[]:
                temp.append(1)
                continue
            flag = self.isBlocked(v)
            if flag == 1:
                temp_i = Runnings.index(requestList[k])
                Runnings.pop(temp_i)
                temp.append(0)
                continue
            else:
                for i in range(len(v) - 1):
                    for e in self.E:
                        if {v[i],v[i+1]} == {e[0],e[1]}:
                            e[4] += 1
                            break
                temp.append(1)

        return temp

    def release(self,releasePath):
        for l in releasePath:
            for i in range(len(l)-1):
                for e in self.E:
                    if {l[i], l[i + 1]} == {e[0], e[1]}:
                        e[4] -= 1
                        break


class Job:
    def __init__(self,path,packetrate):
        self.WillRunnings = []
        self.Running = []
        self.PACKETRATE = packetrate
        self.RATE = round(1/packetrate,6)
        self.drop = 0

        with open(path) as file:
            for data in file:
                datas = data.strip().split()
                self.drop += 1
                finalT = round(int(float(datas[3]) * self.PACKETRATE) * round(self.RATE, 6) + float(datas[0]  ), 6)

                self.WillRunnings.append([datas[1], datas[2], float(datas[0]), float(datas[3]), float(datas[0]), finalT,1, []])
        file.close()


    #检查是否存在即将开始的任务
    def willRunning(self,CurrentTime):
        temp = []
        for j in self.WillRunnings:
            if j[2] <= CurrentTime:
                temp.append(j)
        for j in temp:
            if j == self.WillRunnings[0]:
                self.WillRunnings.pop(0)
        return temp
    #检查是否存在结束的任务
    def hasFinish(self):
        temp = []
        for p in self.Running:
            if p[5] < p[4]:
                temp.append(p)
        for p in temp:
            self.Running.pop(self.Running.index(p))
        return temp
    #更新任务信息
    def update(self,rate):
        for p in self.Running:
            p[4] = round(rate + p[4], 6)


class Manager:
    def __init__(self,network,routing,topology,workload,rate,sources,jobs):

        self.PACKETRATE = int(rate)
        self.Rate_n = 1 / self.PACKETRATE
        self.NETWORKSCHEME = network
        self.ROUTINGSCHEME = routing
        self.workload = workload
        self.topology = topology


        #所有节点信息
        self.Nodes = []
        #存储图
        self.G = []
        #总的delay
        self.Delay = 0
        #总的hops
        self.Hops = 0
        #总的packets
        self.Packets = 0
        #总的circuit
        self.Circuits = 0
        #所有blocketed的
        self.Blockeds = 0
        #blocked的circuit
        self.BlockedsCircuit = 0
        #成功的packet
        self.PacketsSuccess = 0
        self.CircuitsSuccess = 0
        self.MAX = 10000
        #资源管理对象
        self.sources = sources
        #任务管理对象
        self.jobs = jobs
        #请求的任务
        self.requestJob = None
        #释放的任务
        self.releaseJob = None
        self.dijkstra = None


        #初始化topology里的所有节点

        #Node topology里的没一行数据

        with open(self.topology) as file:
            for data in file:
                datas = data.strip().split()
                if datas[0] not in self.Nodes:
                    self.Nodes.append(datas[0])
                if datas[1] not in self.Nodes:
                    self.Nodes.append(datas[1])
        file.close()

        self.run = self.run_packet

        #如果是CIRCUIT模式，那么
        if self.NETWORKSCHEME == 'CIRCUIT':
            self.requestJob = self.requestByCircuit
            self.releaseJob = self.releaseByCircuit
            self.run = self.run_circuit
        elif self.NETWORKSCHEME == 'PACKET':
            self.Circuits -= self.jobs.drop
            self.Packets -= self.jobs.drop
            self.PacketsSuccess -= self.jobs.drop
            self.requestJob = self.requestByPacket
            self.releaseJob = self.releaseByPacket


        # 初始化 图
        for i in range(len(self.Nodes)):
            self.G.append([])
            for j in range(len(self.Nodes)):
                if i == j:
                    self.G[i].append(0)
                else:
                    self.G[i].append(self.MAX)

        # 初始化 图的具体数据
        if self.ROUTINGSCHEME == 'SHP':
            for data in self.sources.E:
                r = self.Nodes.index(data[0])
                c = self.Nodes.index(data[1])
                self.G[r][c] = 1
                self.G[c][r] = 1
            self.dijkstra = self.SHPOrSDP

        elif self.ROUTINGSCHEME == 'LLP':

            for data in self.sources.E:
                r = self.Nodes.index(data[0])
                c = self.Nodes.index(data[1])
                self.G[r][c] = data[2]
                self.G[c][r] = data[2]

            self.dijkstra = self.LLP

        elif self.ROUTINGSCHEME == 'SDP':
            for data in self.sources.E:
                r = self.Nodes.index(data[0])
                c = self.Nodes.index(data[1])
                self.G[r][c] = data[2]
                self.G[c][r] = data[2]

            self.dijkstra = self.SHPOrSDP
    #获取到即将需要发送请求的任务有那些，在packet模式下
    def requestByPacket(self,paths):
        request_temp = []
        for l in self.jobs.Running:
            request_temp.append(l)
        return request_temp
    #在circuit模式下，即将发送的请求
    def requestByCircuit(self,paths):
        return paths
    #circuit下即将释放的circuit
    def releaseByCircuit(self,paths):
        return paths
    #即将释放的任务在packet下
    def releaseByPacket(self,path):
        for l in self.jobs.Running:
            path.append(l[-1])
        return path
    #circuit模式下进行更新
    def UpdateByCircuit(self,path):
        return int(self.PACKETRATE* path[3])
    #获取到当前的延迟和hops
    def getDelay(self,paths):
        temp = paths[-1]
        for i in range(len(temp) - 1):
            for e in self.sources.E:
                if {temp[i],temp[i+1]} == {e[0],e[1]}:
                    self.Delay += e[2]
        self.Hops += (len(temp) - 1)
    #对llp下每次动态更新图
    def costsOfLLP(self):

        for l in self.sources.E:
            r = self.Nodes.index(l[0])
            c = self.Nodes.index(l[1])

            rate = float(l[4]) / l[3]
            self.G[r][c] = rate
            self.G[c][r] = rate
    #根据得到的path stack，获取完整路径，此处由标准dijkstra算法得到的 path stack是一个双亲树
    def getShortestPath(self,stack,nodeS,nodeE):
        path = []
        ns = self.Nodes.index(nodeS)
        ne = self.Nodes.index(nodeE)

        path.append(ne)
        while True:
            value = stack[ne]
            ne = value
            path.append(ne)
            if value == ns:
                break

        tempPath = []
        for v in path:
            tempPath.append(self.Nodes[v])
        tempPath.reverse()

        return tempPath

    #llp的dijisktra算法
    def LLP(self,nodeS,nodeE):
        #重新计算图结构
        self.costsOfLLP()
        #获取到开始节点在Nodes的角标
        ns = self.Nodes.index(nodeS)
        #初始化nods set，用于存放当前已确认最短距离的节点
        ALLNodesSet = [0 for x in self.Nodes]
        #初始化all nodes set
        ALLNodesSet[ns] = 1
        #创建distance 列表，用于存放当前开始节点到达每一个点的最短距离，后续步骤将要一步一步更新该列表
        distanceOfAll = copy.deepcopy(self.G[ns])
        #创建paths列表，用于存放开始节点到达每一个节点的路径，存放的是一个标准双亲树
        paths = []
        #根据NOdes列表，初始化paths， 如果节点和开始节点不相邻，那么设为-1，否则把节点存放进去
        for v in self.Nodes:
            if self.G[ns][self.Nodes.index(v)] == self.MAX:
                paths.append(-1)
            else:
                paths.append(ns)

        #开始dijisktra算法
        for i in range(len(self.Nodes) - 1):
            currentMin = self.MAX
            CurrentMinNode = 0
            #先在所有节点里寻找 开始节点到任意一个节点的最短的距离的节点
            for j in range(len(self.Nodes)):
                if ALLNodesSet[j] == 0 and distanceOfAll[j] < currentMin:
                    CurrentMinNode = j
                    currentMin = distanceOfAll[j]
            #把该节点存入Nodes set，表示已确认
            ALLNodesSet[CurrentMinNode] = 1
            #判断刚确认的节点存入nodesset后，是否会导致当前distance 列表的其中一些距离发生变化，如果发生变化，那么进行更新
            for j in range(len(self.Nodes)):
                temp = max(distanceOfAll[CurrentMinNode], self.G[CurrentMinNode][j])
                if ALLNodesSet[j] == 0 and temp < distanceOfAll[j]:
                    distanceOfAll[j] =temp
                    paths[j] = CurrentMinNode


        #返回paths 列表，里的开始节点到结束节点的最短路径
        return self.getShortestPath(paths,nodeS,nodeE)

    #同上
    def SHPOrSDP(self,nodeS,nodeE):

        ns = self.Nodes.index(nodeS)

        ALLNodesSet = [0 for x in self.Nodes]
        ALLNodesSet[ns] = 1
        distanceOfAll = copy.deepcopy(self.G[ns])

        paths = []
        for v in self.Nodes:
            if self.G[ns][self.Nodes.index(v)] == self.MAX:
                paths.append(-1)
            else:
                paths.append(ns)

        for i in range(len(self.Nodes) - 1):
            currentMin = self.MAX
            CurrentMinNode = 0
            for j in range(len(self.Nodes)):
                if ALLNodesSet[j] == 0 and distanceOfAll[j] < currentMin:
                    CurrentMinNode = j
                    currentMin = distanceOfAll[j]

            ALLNodesSet[CurrentMinNode] = 1

            for j in range(len(self.Nodes)):
                #唯一不同是，上面那个是通过比较大小更新，这个是累加之后更新
                temp = distanceOfAll[CurrentMinNode] + self.G[CurrentMinNode][j]
                if ALLNodesSet[j] == 0 and temp < distanceOfAll[j]:
                    distanceOfAll[j] = temp
                    paths[j] = CurrentMinNode

        p = self.getShortestPath(paths, nodeS, nodeE)


        return self.getShortestPath(paths, nodeS, nodeE)

    #circuit下，经过dijikstra算法后，需要进行资源申请并发送的请求和路径
    def willAcquire(self,requests):
        waite_v = [] #需要发送的请求的任务列表
        waite_request = [] #需要发送请求的任务的路径的列表
        for k, t in enumerate(requests):
            ki = self.jobs.Running.index(t)
            '''
            if t[4] == t[5]:
                self.Circuits += 1
                self.jobs.Running[ki][-1] = []
                continue
            '''

            path = self.dijkstra(t[0], t[1]) #进行dijikstra算法

            self.jobs.Running[ki][-1] = path # 更新每个任务当前的发送路径
            # 等待计算的任务的起始节点
            waite_request.append(path)
            waite_v.append(self.jobs.Running[ki])

            self.Circuits += 1  # 记录circuit树
            # 记录total packet number
            self.Packets += self.UpdateByCircuit(t)

        return [waite_request,waite_v]

    #circuit下进行统计
    def statistics(self,acquireResult,acquirePaths):
        for k, i in enumerate(acquireResult):

            t = acquirePaths[k]

            if i == 0:
                self.BlockedsCircuit += 1
                self.Blockeds += self.UpdateByCircuit(t)
            else:
                self.getDelay(t)
                self.PacketsSuccess += self.UpdateByCircuit(t)

    #查看是否存在已经完成的任务
    def finishPaths(self):
        finishs= self.jobs.hasFinish()
        finishPath = []
        for l in finishs:
            finishPath.append(l[-1])
        #如果存在完成的任务，那么进行释放
        finishPath = self.releaseJob(finishPath)
        return finishPath

    #查看是否结束
    def isCompleted(self):
        if self.jobs.WillRunnings == [] and self.jobs.Running == []:
            return 1
        else:
            return 0

    #对正在运行的任务列表进行更新
    def sortRunning(self,paths):
        if paths != []:
            self.jobs.Running.extend(paths)
            self.jobs.Running.sort(key=lambda x: x[4])


    #packet下的run函数
    def run_packet(self):

        CurrentTime = 0


        while True:

            temp = self.jobs.willRunning(CurrentTime)
            self.sortRunning(temp)

            request_temp = self.requestJob(temp)

            for l in request_temp:

                ki = self.jobs.Running.index(l)
                tl = self.jobs.Running[ki]
                if l[4] == l[2] or tl[6] == 0:
                    # 刚开始的新任务 或者 该任务被block了 ，那么不进行释放
                    pass
                else:
                    rl = l[-1]
                    self.sources.release([rl])

                # 开始申请发送
                self.Packets += 1
                self.Circuits += 1

                path = self.dijkstra(l[0], l[1])
                #更新任务的path
                tl[-1] = path

                flag = self.sources.isBlocked(path)

                #如果block了，那么
                if flag == 1:
                    self.Blockeds += 1
                    tl[6] = 0
                #如果没有block 那么l[6]=1 就是任务没有被block的标志
                else:
                    tl[6] = 1
                    self.PacketsSuccess += 1
                    self.CircuitsSuccess += 1
                    #为该路径请求资源
                    self.sources.acquireLink(path)
                    #计算延时和hop
                    self.getDelay(l)

            # 对每个任务已经执行的时间进行更新
            self.jobs.update(self.Rate_n)
            # 更新当前时间
            CurrentTime = round(CurrentTime + self.Rate_n, 6)

            # 存在即将结束的任务？
            finish_temp = self.jobs.hasFinish()
            finish_path = []
            for l in finish_temp:
                #如果该请求上一次被block了，那不进行释放
                if l[6] == 0:
                    continue
                finish_path.append(l[-1])

            # 对每个结束的路径进行释放资源
            self.sources.release(finish_path)

            # 判断所有任务完成
            if self.isCompleted():
                break

            #这句话可以注释掉，只是为了方便观察运行到哪了
            print(CurrentTime)

        print('total number of virtual circuit requests:', self.Circuits,end='\n\n')
        print('total number of packets:', self.Packets,end='\n\n')
        print('number of successfully routed packets:',self.PacketsSuccess,end='\n\n')
        print('percentage of successfully routed packets:', round(float(self.PacketsSuccess) * 100 / self.Packets, 6)," %",end='\n\n')
        print('number of blocked packets:', self.Blockeds,end='\n\n')
        print('percentage of blocked packets:', round(float(self.Blockeds) * 100 / self.Packets, 6)," %",end='\n\n')
        print('average number of hops per circuit:', round(float(self.Hops) / self.Circuits, 2),end='\n\n')
        print('average cumulative propagation delay per circuit:', round(float(self.Delay) / self.Circuits, 2),end='\n\n')


    def run_circuit(self):

        CurrentTime = 0

        #self.Circuits += self.jobs.drop
        #self.PacketsSuccess += self.jobs.drop


        # 开始执行任务
        while True:
            # 检查是否存在到达的任务
            temp = self.jobs.willRunning(CurrentTime)
            self.sortRunning(temp)

            request_temp = self.requestJob(temp)
            # 记录 需要进行circuit申请的任务（dijkstra算法处理）
            tempAcquire = self.willAcquire(request_temp)

            # 对获得到的最短路径上的每一个link申请资源
            acquireResult = self.sources.acquire(tempAcquire[0], tempAcquire[1],self.jobs.Running)
            #统计
            self.statistics(acquireResult,request_temp)

            # 对每个任务已经执行的时间进行更新
            self.jobs.update(self.Rate_n)
            # 更新当前时间
            CurrentTime = round(CurrentTime + self.Rate_n, 6)

            # 检查是否存在结束的任务
            finishPath = self.finishPaths()
            # 对每个结束的路径进行释放资源
            self.sources.release(finishPath)

            # 判断所有任务完成
            if self.isCompleted():
                break

            # 这句话可以注释掉，只是为了方便观察运行到哪了
            #print(CurrentTime)


        print('total number of virtual circuit requests:', self.Circuits,end='\n\n')
        print('total number of packets:', self.Packets,end='\n\n')
        print('number of successfully routed packets:',self.PacketsSuccess,end='\n\n')
        print('percentage of successfully routed packets:', round(float(self.PacketsSuccess) * 100 / self.Packets, 2)," %",end='\n\n')
        print('number of blocked packets:', self.Blockeds,end='\n\n')
        print('percentage of blocked packets:', round(float(self.Blockeds)  * 100/ self.Packets , 2)," %",end='\n\n')
        print('average number of hops per circuit:', round(float(self.Hops) / self.Circuits, 2),end='\n\n')
        print('average cumulative propagation delay per circuit:', round(float(self.Delay) / self.Circuits, 2),end='\n\n')



if __name__ == "__main__":

    network = sys.argv[1]
    routing = sys.argv[2]
    topology = sys.argv[3]
    workload = sys.argv[4]
    rate = sys.argv[5]

    sources = Source(topology)
    jobs = Job(workload,int(rate))

    manager = Manager(network,routing,topology,workload,rate,sources,jobs)
    manager.run()


