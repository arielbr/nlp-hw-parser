#!/usr/bin/env python
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple

# TODO: Remove
import sys


# logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    args = parser.parse_args()
    return args


class EarleyChart:
    """A chart for Earley's algorithm."""

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self.prune_level = 400

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():  # the last column
            if (item and item.rule.lhs == self.grammar.start_symbol  # a ROOT item in this column
                    and item.next_symbol() is None  # that is complete
                    and item.start_position == 0):  # and started back at position 0
                return True
        return False  # we didn't find any appropriate item

    def print_item(self, item: Item) -> str:
        """
        Print the best parse in the chart.
        :return: The parse of the sentence.
        """

        res = ""

        if item.left_ptr:
            res += self.print_item(item.left_ptr)

        if item.dot_position > 0:  # Avoid empty rhs
            prev = item.rule.rhs[item.dot_position - 1]
            if self.grammar.is_nonterminal(prev):
                res += " (" + prev if not res or res[-1] != "(" else "(" + prev + " "
            else:
                res += " " + prev

        if item.right_ptr:
            res += self.print_item(item.right_ptr) + ")"

        return res

    def _run_earley(self) -> None:
        """Fill in the Earley chart"""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)  # The root item backpoints to (-1, -1)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            logging.debug("")
            logging.debug(f"Processing items in column {i}")
            while column:  # while agenda isn't empty
                item = column.pop()  # dequeue the next unprocessed item
                next = item.next_symbol()
                if next is None:
                    # Attach this complete constituent to its customers
                    logging.debug(f"{item} => ATTACH")
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    # logging.debug(f"{item} => PREDICT")
                    self._predict(next, i)
                else:
                    # Try to scan the terminal after the dot
                    # logging.debug(f"{item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):
            # Compute the new weight
            new_weight = rule.weight

            new_item = Item(rule, dot_position=0, start_position=position, weight=new_weight)
            self.cols[position].push(new_item)
            # logging.info(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced(item)
            self.cols[position + 1].push(new_item)
            # logging.info(f"\tScanned to get: {new_item} in column {position + 1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """

        mid = item.start_position  # start position of this item = end position of item to its left
        for customer in self.cols[mid].all():  # could you eliminate this inefficient linear search?
            if customer and customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced_attach(item)
                self.profile["ATTACH"] += 1
                self.cols[position].push(new_item)


class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.
    """

    def __init__(self) -> None:
        self._items: List[Item] = []  # list of all items that were *ever* pushed
        self._next = 0  # index of first item that has not yet been popped
        self._index: Dict[Item, int] = {}  # stores index of an item if it has been pushed before

        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def push(self, item: Item, prune_level=1e99) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        if item.weight > prune_level:
            return

        if item not in self._index:  # O(1) lookup in hash table
            self._items.append(item)
            self._index[item] = len(self._items) - 1
        else:
            old_item_index = self.get_row_index(item)
            old_item = self.get(old_item_index)

            # New item has lower weight thus replaces the old item or re-process
            if item.weight < old_item.weight:
                next_index = self.next_index()
                if next_index > old_item_index:
                    # MOVE the old item down to allow re-processing
                    self._items[old_item_index] = None  # Remove the old item
                    self._items.append(item)  # Append the new item
                    self._index[item] = len(self._items) - 1  # Reset the item's index
                else:
                    # Overwrite if otherwise
                    self._items[old_item_index] = item

    def next_index(self) -> int:
        """Return the item's index that is being processed currently"""
        return self._next

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        if len(self) == 0:
            raise IndexError
        item = self._items[self._next]
        self._next += 1
        return item

    def get(self, index: int) -> Item:
        """Returns the retrieved item, O(1)"""
        return self._items[index]

    def get_row_index(self, item: Item) -> Item:
        """Returns the index of an item, O(1)"""
        return self._index.get(item)

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items

    def __repr__(self):
        """Provide a REPResentation of the instance for printing."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}  # maps each LHS to the list of rules that expand it
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    @classmethod
    def reduce_terminals_not_appearing(cls, gr: Grammar, sentences: Path, *grammarfiles: Path) -> Grammar:
        """Remove rules with terminals not appearing in the text"""
        words_set = set()
        temp_gr = Grammar(gr.start_symbol, *grammarfiles)
        new_gr = Grammar(gr.start_symbol, *[])  # make new grammar with temporarily empty set of rules
        with open(sentences) as f:
            for sentence in f.readlines():
                if sentence != "":
                    words = sentence.split()
                    words_set = words_set.union(words)
        while 1:
            appeared_non_terms = set()
            count_bef = sum(len(temp_gr._expansions[i]) for i in temp_gr._expansions)
            for lhs in temp_gr._expansions:
                for rule in temp_gr._expansions[lhs]:
                    remove = False
                    for word in rule.rhs:
                        # for deleted 'nonterminals' in subsequent passes, they
                        # never appear as the lhs of any rules, and won't enter this line to be added
                        if not temp_gr.is_nonterminal(word) and word not in words_set:
                            remove = True
                            break
                    if not remove:
                        appeared_non_terms.add(rule.lhs)
                        if rule.lhs not in new_gr._expansions:
                            new_gr._expansions[lhs] = [rule]
                        else:
                            new_gr._expansions[lhs] += [rule]
            count_after = sum(len(new_gr._expansions[i]) for i in new_gr._expansions)
            # no more removals from this recursion
            if count_after == count_bef:
                return new_gr
            temp_gr = new_gr
            new_gr = Grammar(gr.start_symbol, *[])

    @classmethod
    def prune(cls, gr: Grammar, prune_level: float) -> Grammar:
        """prune out grammar rules with prob less than the threshold"""
        new_gr = Grammar(gr.start_symbol, *[])
        new_gr._expansions: Dict[str, List[Rule]] = {}  # maps each LHS to the list of rules that expand it
        for lhs in gr._expansions:
            for rule in gr._expansions[lhs]:
                if rule.weight <= prune_level:
                    if rule.lhs not in new_gr._expansions:
                        new_gr._expansions[lhs] = [rule]
                    else:
                        new_gr._expansions[lhs] += [rule]
        return new_gr

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    Convenient abstraction for a grammar rule. 
    A rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"


# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse table, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    weight: float  # Note that the weight here is already -log2prob
    left_ptr: Item = None
    right_ptr: Item = None

    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    """See https://stackoverflow.com/questions/2909106/whats-a-correct-and-good-way-to-implement-hash for building 
    partial hash"""

    def __key(self):
        return (self.rule, self.dot_position, self.start_position)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Item):
            return self.__key() == other.__key()
        return False

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self, item: Item) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position,
                    weight=self.weight, left_ptr=item)

    def with_dot_advanced_attach(self, item: Item):
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, start_position=self.start_position,
                    weight=self.weight + item.weight, left_ptr=self, right_ptr=item)

    def __repr__(self) -> str:
        """Complete string used to show this item at the command line"""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        # return f"{self.weight:.2f} {self.start_position}, {dotted_rule}"
        return f"{self.weight:.2f} {self.start_position}, {dotted_rule} | left: ({self.left_ptr}) | right(" \
               f"{self.right_ptr})"


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.verbose)  # Set logging level appropriately

    grammar = Grammar(args.start_symbol, args.grammar)
    # grammar = Grammar.reduce_terminals_not_appearing(grammar, args.sentences, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                prune_level_max = 800
                # analyze the sentence
                found = False
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                while chart.prune_level <= prune_level_max:
                    chart._run_earley()
                    # print the result
                    logging.debug(f"Profile of work done: {chart.profile}")
                    if chart.accepted():
                        last_item = None
                        last_weight = 1e99
                        for item in chart.cols[-1].all():  # the last column
                            if (item and item.rule.lhs == chart.grammar.start_symbol  # a ROOT item in this column
                                    and item.next_symbol() is None  # that is complete
                                    and item.start_position == 0  # and started back at position 0
                                    and item.weight < last_weight):  # has minimal weight
                                last_item = item
                                last_weight = item.weight
                        s = chart.print_item(last_item).strip()
                        s = "(" + args.start_symbol + " " + s + ")"
                        print(s)
                        print(last_item.weight)
                        found = True
                        break
                    chart.prune_level += 100
                if not found:
                    print("NONE")


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)  # run tests
    main()
