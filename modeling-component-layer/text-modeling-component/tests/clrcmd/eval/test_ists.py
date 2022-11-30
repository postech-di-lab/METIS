import pytest

from clrcmd.eval.ists import load_instances


@pytest.fixture
def filepath_sent1(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent1.txt")
    tmpfile.write(
        """both bulbs a and c still have a closed path
terminal 1 and the positive terminal are connected.
positive battery is seperated by a gap from terminal 2
There is no difference between the two terminals.
the switch has to be contained in the same path as the bulb and the battery"""
    )
    return tmpfile.strpath


@pytest.fixture
def filepath_sent2(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent2.txt")
    tmpfile.write(
        """Bulbs A and C are still in closed paths
Terminal 1 and the positive terminal are separated by the gap
Terminal 2 and the positive terminal are separated by the gap
The terminals are in the same state.
The switch and the bulb have to be in the same path."""
    )
    return tmpfile.strpath


@pytest.fixture
def filepath_sent1_chunk(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent1.chunk.txt")
    tmpfile.write(
        """[ both ] [ bulbs a and c ] [ still ] [ have ] [ a closed path ]
[ terminal 1 and the positive terminal ] [ are connected. ]
[ positive battery ] [ is seperated ] [ by a gap ] [ from terminal 2 ]
[ There ] [ is ] [ no difference ] [ between the two terminals. ]
[ the switch ] [ has to be contained ] [ in the same path ] [ as ] [ the bulb and the battery ]"""
    )
    return tmpfile.strpath


@pytest.fixture
def filepath_sent2_chunk(tmpdir):
    tmpfile = tmpdir.join("STSint.testinput.answers-students.sent2.chunk.txt")
    tmpfile.write(
        """[ Bulbs A and C ] [ are ] [ still ] [ in closed paths ]
[ Terminal 1 and the positive terminal ] [ are separated ] [ by the gap ]
[ Terminal 2 and the positive terminal ] [ are separated ] [ by the gap ]
[ The terminals ] [ are ] [ in the same state. ]
[ The switch and the bulb ] [ have to be ] [ in the same path. ]"""
    )
    return tmpfile.strpath


def test_load_instances(
    filepath_sent1, filepath_sent2, filepath_sent1_chunk, filepath_sent2_chunk
):
    pred = load_instances(
        filepath_sent1, filepath_sent2, filepath_sent1_chunk, filepath_sent2_chunk
    )
    true = [
        {
            "id": 1,
            "sent1": "both bulbs a and c still have a closed path",
            "sent2": "Bulbs A and C are still in closed paths",
            "sent1_chunk": ["both", "bulbs a and c", "still", "have", "a closed path"],
            "sent2_chunk": ["Bulbs A and C", "are", "still", "in closed paths"],
        },
        {
            "id": 2,
            "sent1": "terminal 1 and the positive terminal are connected.",
            "sent2": "Terminal 1 and the positive terminal are separated by the gap",
            "sent1_chunk": ["terminal 1 and the positive terminal", "are connected."],
            "sent2_chunk": [
                "Terminal 1 and the positive terminal",
                "are separated",
                "by the gap",
            ],
        },
        {
            "id": 3,
            "sent1": "positive battery is seperated by a gap from terminal 2",
            "sent2": "Terminal 2 and the positive terminal are separated by the gap",
            "sent1_chunk": [
                "positive battery",
                "is seperated",
                "by a gap",
                "from terminal 2",
            ],
            "sent2_chunk": [
                "Terminal 2 and the positive terminal",
                "are separated",
                "by the gap",
            ],
        },
        {
            "id": 4,
            "sent1": "There is no difference between the two terminals.",
            "sent2": "The terminals are in the same state.",
            "sent1_chunk": [
                "There",
                "is",
                "no difference",
                "between the two terminals.",
            ],
            "sent2_chunk": ["The terminals", "are", "in the same state."],
        },
        {
            "id": 5,
            "sent1": (
                "the switch has to be contained in the same path as the bulb and the" " battery"
            ),
            "sent2": "The switch and the bulb have to be in the same path.",
            "sent1_chunk": [
                "the switch",
                "has to be contained",
                "in the same path",
                "as",
                "the bulb and the battery",
            ],
            "sent2_chunk": [
                "The switch and the bulb",
                "have to be",
                "in the same path.",
            ],
        },
    ]
    assert pred == true
