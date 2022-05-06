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


class Early(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.s == 1:
            self.FSM.to_transition("toContact")
        return str("Early")


class Contact(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.s == 0:
            self.FSM.to_transition("toFlight")
        return str("Contact")


class Late(State):
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        if self.FSM.sh == 1:
            self.FSM.to_transition("toContact")
        return str("Late")


class Flight(State):
    # Aerial Phase
    def __init__(self, fsm):
        super().__init__(fsm)

    def execute(self):
        # if self.FSM.sh == 1:  # and self.FSM.pdot[2] <= 0:
        if self.FSM.sh == 1 and self.FSM.s == 0:
            self.FSM.to_transition("toEarly")

        if self.FSM.s == 1 and self.FSM.sh == 1:
            self.FSM.to_transition("toContact")

        if self.FSM.sh == 0 and self.FSM.s == 1:
            self.FSM.to_transition("toLate")

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

    def execute(self, s, sh):
        self.s = s
        self.sh = sh
        
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
        self.Contact = True

        self.FSM.add_state("Contact", Contact(self.FSM))
        self.FSM.add_state("Flight", Flight(self.FSM))
        self.FSM.add_state("Early", Early(self.FSM))
        self.FSM.add_state("Late", Late(self.FSM))

        self.FSM.add_transition("toContact", Transition("Contact"))
        self.FSM.add_transition("toFlight", Transition("Flight"))
        self.FSM.add_transition("toEarly", Transition("Early"))
        self.FSM.add_transition("toLate", Transition("Late"))

        self.FSM.setstate("Flight")

    def execute(self):
        self.FSM.execute(s, sh)
