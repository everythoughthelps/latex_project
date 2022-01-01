import numpy as np


class DiscountProblem():
    """
    折扣系数解订单问题，实现了价值迭代和策略迭代
    """

    def __init__(self, c, K, n, p, a):
        self.c = c
        self.K = K
        self.n = n
        self.p = p
        self.a = a
        self.action_prob = {0: 0.5, 1: 0.5}
        self.transition = self.__init_transition()
        self.V = [0 for _ in range(n + 1)]

    def __init_transition(self):
        state_action = {}
        # 序列状态为{0,1,...n},转移矩阵定义依据PPT
        for i in range(self.n + 1):
            if (i == self.n):
                state_action[self.n] = {1: {1: self.p, 0: 1 - self.p}}
                continue
            state_action[i] = {1: {0: 1 - self.p, 1: self.p}, 0: {i + 1: self.p, i: 1 - self.p}}
        return state_action

    def next_best_action(self, s, V):
        action_values = np.zeros(2)
        for a in self.transition[s]:
            for state in self.transition[s][a]:
                cost = self.K if a == 1 else self.c * s
                action_values[a] += self.transition[s][a][state] * (cost + self.a * V[state])
        return np.argmin(action_values), np.min(action_values)

    def value_Iteration(self):
        """
        价值迭代算法实现
        :return: 最优策略
        """
        THETA = 0.0001
        delta = float("inf")
        round_num = 0

        while delta > THETA:
            print(delta)
            delta = 0
            print("\nValue Iteration: Round " + str(round_num))
            for s in range(self.n + 1):
                best_action, best_action_value = self.next_best_action(s, self.V)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                self.V[s] = best_action_value

            print(delta)
            round_num += 1

        # policy = np.zeros(self.n + 1)
        policy=[]
        for s in range(self.n + 1):
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy.append(best_action)
            # policy[s] = best_action
        return policy

    def __policy_evaluation(self):
        """
        策略评估
        :return: 收敛的值函数
        """
        V = np.zeros(self.n+1)
        THETA = 0.0001
        delta = float("inf")

        while delta > THETA:
            delta = 0
            for s in range(self.n+1):
                expected_value = 0
                for a in self.transition[s]:
                    for state in self.transition[s][a]:
                        cost = self.K if a == 1 else self.c * s
                        # print("state_prob",self.transition[s][a][state])
                        expected_value += 0.5*self.transition[s][a][state] * (cost + self.a * V[state])
                # print(expected_value)
                delta = max(delta, np.abs(V[s] - expected_value))
                V[s] = expected_value
        return V

    def policy_iteration(self):
        policy = np.tile(np.eye(2)[1], (self.n+1, 1))

        is_stable = False

        round_num = 0

        while not is_stable:
            is_stable = True

            print("\nRound Number:" + str(round_num))
            round_num += 1

            print("Current Policy")

            V = self.__policy_evaluation()
            # print("Expected Value accoridng to Policy Evaluation")
            # print(np.reshape(V, self.env.shape))

            for s in range(self.n+1):
                action_by_policy = np.argmax(policy[s])
                best_action, best_action_value = self.next_best_action(s, V)
                # print("\nstate=" + str(s) + " action=" + str(best_action))
                policy[s] = np.eye(2)[best_action]
                if action_by_policy != best_action:
                    is_stable = False

        policy = [np.argmax(policy[s]) for s in range(self.n+1)]
        return policy


if __name__ == '__main__':
    print(np.eye(2)[1])
    print(np.tile(np.eye(2)[1], (5, 1)))
    DiscountProblem(5, 12, 40, 0.4, 0.7).policy_iteration()
