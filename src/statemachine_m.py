"""
Copyright (C) 2020 Benjamin Bokser
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


class Compress(State):
    # First Half of Contact Phase
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.pdot[2] >= 0:
            self.FSM.to_transition("toPush")
        return str("Compress")


class Push(State):
    # Second Half of Contact Phase
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.sh == 0 and self.FSM.s == 0:  #self.FSM.leg_pos[2] < -self.FSM.h_tran:
            self.FSM.to_transition("toRise")
        # print(self.FSM.leg_pos[2])
        return str("Push")


class Rise(State):
    # First Half of Aerial Phase
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.pdot[2] <= 0:  # Recognize that the bot is falling
            self.FSM.to_transition("toFall")

        return str("Rise")


class Fall(State):
    # Second Half of Aerial Phase
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.sh == 1:
            self.FSM.to_transition("toCompress")

        return str("Fall")


class Transition:
    def __init__(self, tostate):
        self.toState = tostate

    def execute(self):
        pass


class FSM:
    def __init__(self, char, h_tran):
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
        self.h_tran = h_tran  # nominal height

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
    def __init__(self, h_tran):
        self.FSM = FSM(self, h_tran)
        self.Push = True

        self.FSM.add_state("Push", Push(self.FSM))
        self.FSM.add_state("Rise", Rise(self.FSM))
        self.FSM.add_state("Fall", Fall(self.FSM))
        self.FSM.add_state("Compress", Compress(self.FSM))

        self.FSM.add_transition("toPush", Transition("Push"))
        self.FSM.add_transition("toRise", Transition("Rise"))
        self.FSM.add_transition("toFall", Transition("Fall"))
        self.FSM.add_transition("toCompress", Transition("Compress"))

        self.FSM.setstate("Fall")

    def execute(self):
        self.FSM.execute(s, sh, go, pdot, leg_pos)
