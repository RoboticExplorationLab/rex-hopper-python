"""
Copyright (C) 2020-2022 Benjamin Bokser
"""


class State:
    def __init__(self, fsm):
        self.FSM = fsm

    def enter(self):
        pass

    def execute(self):

        pass

    def exit(self):
        pass


class Stance(State):
    # Thrust up until unloading
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.sh == 0 and self.FSM.leg_pos[2] < -0.4:
            self.FSM.to_transition("toFlight")
        # print(self.FSM.leg_pos[2])
        return str("Stance")


class Flight(State):
    # Aerial Phase
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        # if self.FSM.sh == 1:  # and self.FSM.pdot[2] <= 0:
        if self.FSM.s == 1 and self.FSM.sh == 1:  # and self.FSM.go == True:  # wait to jump until scheduled to
            self.FSM.to_transition("toStance")

        return str("Flight")


class Transition:
    def __init__(self, tostate):
        self.toState = tostate

    def execute(self):
        pass


class FSM:
    def __init__(self, char):
        self.char = char  # passed in
        self.states = {}
        self.transitions = {}
        self.curState = None
        self.prevState = None
        self.trans = None

        self.s = None
        self.sh = None
        self.go = None  # Prevents stuck-in-stance bug
        self.pdot = None
        self.leg_pos = None

    def add_transition(self, transname, transition):
        self.transitions[transname] = transition

    def add_state(self, statename, state):
        self.states[statename] = state

    def setstate(self, statename):
        # look for whatever state we passed in within the states dict
        self.prevState = self.curState
        self.curState = self.states[statename]

    def to_transition(self, to_trans):
        # set the transition state
        self.trans = self.transitions[to_trans]

    def execute(self, s, sh, go, pdot, leg_pos):
        self.s = s
        self.sh = sh
        self.go = go
        self.pdot = pdot
        self.leg_pos = leg_pos
        
        if self.trans:
            self.curState.exit()
            self.trans.execute()
            self.setstate(self.trans.toState)
            self.curState.enter()
            self.trans = None

        output = self.curState.execute()

        return output


class Char:
    def __init__(self):
        self.FSM = FSM(self)
        self.Stance = True

        self.FSM.add_state("Stance", Stance(self.FSM))
        self.FSM.add_state("Flight", Flight(self.FSM))

        self.FSM.add_transition("toStance", Transition("Stance"))
        self.FSM.add_transition("toFlight", Transition("Flight"))

        self.FSM.setstate("Flight")

    def execute(self):
        self.FSM.execute(s, sh, go, pdot, leg_pos)
