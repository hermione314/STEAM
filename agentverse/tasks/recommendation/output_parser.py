from __future__ import annotations

import re
import json
from typing import Union

from agentverse.parser import OutputParser, LLMResult

# from langchain.schema import AgentAction, AgentFinish
from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import OutputParserError, output_parser_registry


@output_parser_registry.register("recommender")
class RecommenderParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('Choice') + len('Choice:')
        except:
            print(cleaned_output)
            print("!!!!!")
        ans_end = cleaned_output.index('Explanation')
        rat_begin = cleaned_output.index('Explanation') + len('Explanation:')
        ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        if ans == '' or rat == '':
            raise OutputParserError(text)
        return ans, rat

    def parse_backward(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        # ans_begin = cleaned_output.index('Reasons: ') + len('Reasons: ')
        # ans_end = cleaned_output.index('Reflections')
        rat_begin = cleaned_output.index('Updated Strategy') + len('Updated Strategy:')
        # ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        return rat

    def parse_clustered(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        # ans_begin = cleaned_output.index('Reasons: ') + len('Reasons: ')
        # ans_end = cleaned_output.index('Reflections')
        rat_begin = cleaned_output.index('Summary is:') + len('Summary is:')
        # ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        return rat

    def parse_summary(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_evaluation(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        # print(cleaned_output)
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        # return cleaned_output

        try:
            ans_begin = cleaned_output.index('Rank:') + len('Rank:')
        except:
            ans_begin = cleaned_output.index('1.') + len('1.')
            print(cleaned_output)
        # ans_end = cleaned_output.index('Rationale')
        # rat_begin = cleaned_output.index('Rationale') + len('Rationale: ')
        ans = cleaned_output[ans_begin:].strip().split('\n')
        return ans



@output_parser_registry.register("useragent")
class UserAgentParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()


    def parse_summary(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        return cleaned_output.strip()

    def parse_update(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        rat_begin = None
        try:
            rat_begin = cleaned_output.index('My updated personal memory: ') + len('My updated personal memory: ')
        except:
            rat_begin = None
        if rat_begin != None:
            rat = cleaned_output[rat_begin:].strip()
        else:
            rat = cleaned_output.strip()
        return rat
    

    def parse_summary_update(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            rat_begin = cleaned_output.index('Summary User Memory: ') + len('Summary User Memory: ')
        except:
            print(cleaned_output)
        rat = cleaned_output[rat_begin:].strip()
        return rat
    

    def parse_judge_update(self, text: str) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)  # 清除多余的换行符

        # 用正则表达式提取 judgment 和 memory
        try:
            # 提取 judgment（在 'Judgment of memory error:' 后）
            judgment_begin = cleaned_output.index('Memory misjudgment: ') + len('Memory misjudgment: ')
            judgment_end = cleaned_output.index('my updated personal memory: ')
            judgment = cleaned_output[judgment_begin:judgment_end-1].strip()

            # 提取 memory（在 'My updated personal memory:' 后）
            memory_begin = cleaned_output.index('my updated personal memory: ') + len('my updated personal memory: ')
            memory = cleaned_output[memory_begin:].strip()

            # 返回 judgment 和 memory
            return judgment, memory
        except Exception as e:
            print("Error in parsing the text:", e)
            print(cleaned_output)
            return None, None



@output_parser_registry.register("itemagent")
class ItemAgentParser(OutputParser):
    def parse(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        try:
            ans_begin = cleaned_output.index('The updated description of the first CD') + len(
                'The updated description of the first CD') + 4
        except:
            print(cleaned_output)
        try:
            ans_end = cleaned_output.index('The updated description of the second CD')
        except:
            print(cleaned_output)
        rat_begin = cleaned_output.index('The updated description of the second CD') + len(
            'The updated description of the second CD') + 4
        ans = cleaned_output[ans_begin:ans_end].strip()
        rat = cleaned_output[rat_begin:].strip()
        if ans == '' or rat == '':
            raise OutputParserError(text)
        return ans, rat

    def parse_pretrain(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        ans_begin = cleaned_output.index('CD Description: ') + len('CD Description: ')
        ans = cleaned_output[ans_begin:].strip()
        return ans

    def parse_aug(self, text: LLMResult) -> Union[AgentAction, AgentFinish]:
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        ans_begin = cleaned_output.index('Speculated CD Reviews: ') + len('Speculated CD Reviews: ')
        ans = cleaned_output[ans_begin:].strip()
        return ans


