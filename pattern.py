patterns = {
    "ace05-en": {
        "Business:Declare-Bankruptcy": {
            "Org": "What declare bankruptcy in [trigger]?", 
            "Place": "Where the merger takes place in [trigger]?", 
        }, 
        "Business:End-Org": {
            "Place": "Where the event takes place in [trigger]?", 
            "Org": "What is ended in [trigger]?", 
        }, 
        "Business:Merge-Org": {
            "Org": "What is merged in [trigger]?", 
        }, 
        "Business:Start-Org": {
            "Org": "What is started in [trigger]?", 
            "Place": "Where the event takes place in [trigger]?", 
            "Agent": "Who is the founder in [trigger]?", 
        }, 
        "Conflict:Attack": {
            "Place": "Where the attack takes place in [trigger]?", 
            "Target": "Who is the target of the attack in [trigger]?", 
            "Attacker": "Who is the attacking agent in [trigger]?", 
            "Instrument": "What is the instrument used in the attack in [trigger]?", 
            "Victim": "Who is the target of the attack in [trigger]?", 
        }, 
        "Conflict:Demonstrate": {
            "Entity": "Who is demonstrating agent in [trigger]?", 
            "Place": "Where the demonstration takes place in [trigger]?", 
        }, 
        "Contact:Meet": {
            "Entity": "Who are meeting in [trigger]?", 
            "Place": "Where the meeting takes place in [trigger]?", 
        }, 
        "Contact:Phone-Write": {
            "Entity": "Who is communicating agents in [trigger]?", 
            "Place": "Where it takes place in [trigger]?", 
        }, 
        "Justice:Acquit": {
            "Defendant": "Who was acquitted in [trigger]?", 
            "Adjudicator": "Who was the judge or court in [trigger]?", 
        }, 
        "Justice:Appeal": {
            "Adjudicator": "Who was the judge or court in [trigger]?", 
            "Plaintiff": "What is the plaintiff in [trigger]?", 
            "Place": "Where the appeal takes place in [trigger]?", 
        }, 
        "Justice:Arrest-Jail": {
            "Person": "Who is jailed or arrested in [trigger]?", 
            "Agent": "Who is the arresting agent in [trigger]?", 
            "Place": "Where the person is arrested in [trigger]?", 
        }, 
        "Justice:Charge-Indict": {
            "Adjudicator": "Who was the judge or court in [trigger]?", 
            "Defendant": "Who is indicted in [trigger]?", 
            "Prosecutor": "Indicated by whom in [trigger]?", 
            "Place": "Where the indictment takes place in [trigger]?", 
        }, 
        "Justice:Convict": {
            "Defendant": "Who is convicted in [trigger]?", 
            "Adjudicator": "Who is the judge or court in [trigger]?", 
            "Place": "Where the conviction takes place in [trigger]?", 
        }, 
        "Justice:Execute": {
            "Place": "Where the execution takes place in [trigger]?", 
            "Agent": "Who carry out the execution in [trigger]?", 
            "Person": "Who was executed in [trigger]?", 
        }, 
        "Justice:Extradite": {
            "Origin": "Where is original location of the person being extradited in [trigger]?", 
            "Destination": "Where the person is extradited to in [trigger]?", 
            "Agent": "Who is the extraditing agent in [trigger]?", 
        }, 
        "Justice:Fine": {
            "Entity": "What was fined in [trigger]?", 
            "Adjudicator": "Who do the fining in [trigger]?", 
            "Place": "Where the fining Event takes place in [trigger]?", 
        }, 
        "Justice:Pardon": {
            "Adjudicator": "Who do the pardoning in [trigger]?", 
            "Place": "Where the pardon takes place in [trigger]?", 
            "Defendant": "Who was pardoned in [trigger]?", 
        }, 
        "Justice:Release-Parole": {
            "Entity": "Who will release in [trigger]?", 
            "Person": "Who is released in [trigger]?", 
            "Place": "Where the release takes place in [trigger]?", 
        }, 
        "Justice:Sentence": {
            "Defendant": "Who is sentenced in [trigger]?", 
            "Adjudicator": "Who is the judge or court in [trigger]?", 
            "Place": "Where the sentencing takes place in [trigger]?", 
        }, 
        "Justice:Sue": {
            "Plaintiff": "Who is the suing agent in [trigger]?", 
            "Defendant": "Who is sued against in [trigger]?", 
            "Adjudicator": "Who is the judge or court in [trigger]?", 
            "Place": "Where the suit takes place in [trigger]?", 
        }, 
        "Justice:Trial-Hearing": {
            "Defendant": "Who is on trial in [trigger]?", 
            "Place": "Where the trial takes place in [trigger]?", 
            "Adjudicator": "Who is the judge or court in [trigger]?", 
            "Prosecutor": "Who is the prosecuting agent in [trigger]?", 
        }, 
        "Life:Be-Born": {
            "Place": "Where the birth takes place in [trigger]?", 
            "Person": "Who is born in [trigger]?", 
        }, 
        "Life:Die": {
            "Victim": "Who died in [trigger]?", 
            "Agent": "Who is the attacking agent in [trigger]?", 
            "Place": "Where the death takes place in [trigger]?", 
            "Instrument": "What is the device used to kill in [trigger]?", 
        }, 
        "Life:Divorce": {
            "Person": "Who are divorced in [trigger]?", 
            "Place": "Where the divorce takes place in [trigger]?", 
        }, 
        "Life:Injure": {
            "Victim": "Who is the harmed person in [trigger]?", 
            "Agent": "Who is the attacking agent in [trigger]?", 
            "Place": "Where the injuring takes place in [trigger]?", 
            "Instrument": "What is the device used to inflict the harm in [trigger]?", 
        }, 
        "Life:Marry": {
            "Person": "Who are married in [trigger]?", 
            "Place": "Where the marriage takes place in [trigger]?", 
        }, 
        "Movement:Transport": {
            "Vehicle": "What is the vehicle used to transport the person or artifact in [trigger]?", 
            "Artifact": "Who is being transported in [trigger]?", 
            "Destination": "Where the transporting is directed in [trigger]?", 
            "Agent": "Who is responsible for the transport event in [trigger]?", 
            "Origin": "Where the transporting originated in [trigger]?", 
        }, 
        "Personnel:Elect": {
            "Person": "Who was elected in [trigger]?", 
            "Entity": "Who voted in [trigger]?", 
            "Place": "Where the election takes place in [trigger]?", 
        }, 
        "Personnel:End-Position": {
            "Entity": "Who is the employer in [trigger]?", 
            "Person": "Who is the employee in [trigger]?", 
            "Place": "Where the employment relationship ends in [trigger]?", 
        }, 
        "Personnel:Nominate": {
            "Person": "Who are nominated in [trigger]?", 
            "Agent": "Who is the nominating agent in [trigger]?", 
        }, 
        "Personnel:Start-Position": {
            "Person": "Who is the employee in [trigger]?", 
            "Entity": "Who is the employer in [trigger]?", 
            "Place": "Where the employment relationship begins in [trigger]?", 
        }, 
        "Transaction:Transfer-Money": {
            "Giver": "Who is the donating agent in [trigger]?", 
            "Recipient": "Who is the recipient in [trigger]?", 
            "Beneficiary": "Who benefits from the transfer in [trigger]?", 
            "Place": "Where the transaction takes place in [trigger]?", 
        }, 
        "Transaction:Transfer-Ownership": {
            "Buyer": "Who is the buying agent in [trigger]?", 
            "Artifact": "Who was bought or sold in [trigger]?", 
            "Seller": "Who is the selling agent in [trigger]?", 
            "Place": "Where the sale takes place in [trigger]?", 
            "Beneficiary": "Who benefits from the transaction in [trigger]?", 
        }, 
    }
}