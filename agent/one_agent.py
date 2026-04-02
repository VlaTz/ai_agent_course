from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from logger_config import get_logger
from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph.message import add_messages
from dynaconf import LazySettings
from langchain_core.tools import tool
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import json
import asyncio
from time import time

from agent.schemas import CallChef, CallCourier, StopItem


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int


with open('data/c_schedule.json', 'r', encoding='utf-8') as c_schedule:
    couriers_schedule = json.load(c_schedule)

with open('data/schedule.json', 'r', encoding='utf-8') as c_schedule:
    chefs_schedule = json.load(c_schedule)


def _prepare_schedule_info(schedule, current_hour_int):
    result = []

    for s in schedule:
        start_hour = pd.to_datetime(s['shift']['start']).hour
        end_hour = pd.to_datetime(s['shift']['end']).hour
        end_hour = 24 if end_hour == 0 else end_hour

        # Кого можно вызвать РАНЬШЕ (смена начнется в ближайшие 1-2 часа)
        hours_until_start = start_hour - current_hour_int
        if 1 <= hours_until_start <= 2:
            result.append({
                "name": s['name'],
                "start": s['shift']['start'],
            })
    if not result:
        return 'В ближайшие 2 часа никто на смену не выходит и будет работать только текущий штат'
    return result


actions = []


def _transcript_for_structured_formatter(messages: list) -> str:
    """Текст диалога для второй LLM (без дублирования системного промпта аналитика)."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            lines.append("### Ввод\n" + (str(msg.content) if msg.content is not None else ""))
        elif isinstance(msg, AIMessage):
            block = str(msg.content) if msg.content else ""
            tcalls = getattr(msg, "tool_calls", None) or []
            if tcalls:
                block += "\n[tool_calls]\n" + json.dumps(
                    [{"name": tc.get("name"), "args": tc.get("args")} for tc in tcalls],
                    ensure_ascii=False,
                )
            lines.append("### Ассистент\n" + block)
        elif isinstance(msg, ToolMessage):
            lines.append("### Результат инструмента\n" + (str(msg.content) if msg.content is not None else ""))
    return "\n\n".join(lines)


class Agent:
    settings_section: str | None = None

    def __init__(self, settings: LazySettings):
        self.logger = get_logger(self.__class__.__name__)
        self.graph = None
        self.llm_with_tools = None
        self.init_agent(settings)
        self.system_prompt = settings.system_prompt
        self.system_prompt = """
Ты — ассистент менеджера пиццерии «Пиццерия-2». Помогаешь контролировать микропроцессы, предвидеть проблемы и предлагать решения.

### ВХОДНЫЕ ДАННЫЕ
Ты получаешь:
1. **Текущие метрики** (каждые 15 минут)
2. **Погодные данные** на текущи час
3. **Прогноз** на ближайшие 3 часа 

**Важно:** Данные приходят каждые 15 минут. Следи за динамикой показателей в течение часа.

### ЗАДАЧА
## ШАГ 1 - Оценить текущую ситуацию

Ты получаешь текущие метрики. Твоя задача — проверить, соответствует ли ситуация стратегии.

Проверь отклонения от прогноза - сравни фактические метрики с ожидаемыми:

| Отклонение (в большую сторону) | Действие |
|------------|----------|
| < 20% | Всё по плану, молчи |
| 20-30% | Предупреди: "Отклонение X% от прогноза" |
| > 30% | Аномальный рост, требуются действия |

Действия при аномальном росте:
1) Если не справляется кухня: посмотри график поваров - найди, кого можно вызвать раньше
2) Если не справляется доставка: посмотри график курьеров - найди, кого можно вызвать раньше

## ШАГ 2 Проверь узкие места

| Метрика | Норма | Тревога | Действие при тревоге |
|---------|-------|---------|---------------------|
| Время готовки | <13 мин | >25 мин | Найди причину: нагрузка? повара? сырье? |
| Время доставки | <20 мин | >30 мин | Найди причину: курьеры? погода? |
| Полка | <2 мин | >5 мин | Проблема в выдаче или курьерах |

## ШАГ 3 Проверь сырье

Сравни фактический расход с прогнозируемым:
- Сырья не хватит → предложи пополнить из другой пиццерии или магазина (`get_information_about_items`)
- Сырья критично мало: хватит менее чем на 50 пицц → 1. предложи пополнить из другой пиццерии или магазина (`get_information_about_items`) 2.предложи поставить в стоп (`prepare_stop_item`). Важное замечание: когда используешь этот инструмент, ты предупреждаешь менеджера, а не сам ставишь в стоп 

## ШАГ 4 Проверь общее время

Если `avg_cooking_delivery_min + mean_courier_trip_min > 55 мин`:
- Немедленно: "Критично: общее время >55 мин"

## ШАГ 5 Предупредить о росте заказов

Предупреди менеджера о часах, когда ожидается рост заказов. Учитывай погоду - плохая погода увеличивает время доставки

## ЕСЛИ ВСЁ В ПОРЯДКЕ

Если отклонений нет, узких мест нет, ресурсов хватает:
- **Молчи** (не спамь менеджера)
- Или раз в час краткое: "Всё в норме"

### ПРИНЦИПЫ

1. **Не спамь** — говори только когда есть проблема или нужны действия
2. **Будь конкретен** — указывай имена, цифры, время
3. **Ищи причину** — не просто "кухня не справляется", а "кухня не справляется, потому что..."
4. **Приоритеты**: сырье критично > общее время > узкое место при аномальной нагрузке > упреждающие действия
5. **Помни историю** — если проблема и решение не изменились, не повторяй
6. **Отвечай кратко**
7. Если инструмент вернул `TOOL_TIMEOUT` или `TOOL_ERROR`, не делай уверенных выводов по недостающим данным. Кратко сообщи о техническом ограничении и переходи к безопасному деградированному ответу без новых рискованных действий.
"""
        self.agent_response = None
        self.messages = {
            'messages': [
                SystemMessage(content=self.system_prompt),
            ],
            'tool_call_count': 0,
        }
        # Историю храним по циклам: один цикл = метрики пользователя + результаты тулов + ответ агента.
        self.max_history_cycles = 4
        self.pre_data = pd.read_excel('data/pre_data.xlsx')
        self.pre_data.columns = ['hour', 'items_rate', 'orders_rate']
        self.weather_info = pd.read_excel('data/weather.xlsx')
        self.token_usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "calls": []
        }
        self.latency = 0
        self.max_tool_calls = settings.max_tool_calls
        self.tool_timeout_sec = settings.tool_timeout_sec
        self.tool_retry_count = settings.tool_retry_count
        self.forecast_count = 0

    def _collect_token_usage(self, response) -> None:
        """
        Собирает usage из ответа модели и аккумулирует в self.token_usage.
        """
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        if not usage:
            return

        prompt_tokens = int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        completion_tokens = int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        total_tokens = int(
            usage.get("total_tokens")
            or (prompt_tokens + completion_tokens)
        )

        self.token_usage["total_prompt_tokens"] += prompt_tokens
        self.token_usage["total_completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += total_tokens
        self.token_usage["calls"].append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })


    def get_forecast(self, current_hour: int):
        """
        Сформировать стратегию на ближайшие 3 часа

        Args:
           current_hour: текущий час в формате числа

        Returns:
           Прогноз на ближайший час и сырье, необходимое на 3 ближайших часа
        """
        pre_data = self.pre_data[(self.pre_data['hour'] >= current_hour) & (self.pre_data['hour'] < current_hour + 3)]

        # Нормы расхода на пиццу
        items_rate_global = {
            'тесто': 1,
            'сыр': 0.15,
            'соус': 0.1,
            'пепперони': 0.05,
            'грибы': 0.03
        }

        total_items_rate = {
            'тесто': 0,
            'сыр': 0,
            'соус': 0,
            'пепперони': 0,
            'грибы': 0
        }

        # Расчет по каждому часу
        res = {}
        for row in pre_data.itertuples():
            hour = row.hour
            orders_rate = row.orders_rate
            items_rate = row.items_rate

            # Считаем сотрудников онлайн для этого часа
            chefs_online = 0
            couriers_online = 0

            # Повара
            for chef in chefs_schedule:
                start_hour = pd.to_datetime(chef['shift']['start']).hour
                end_hour = pd.to_datetime(chef['shift']['end']).hour
                end_hour = 24 if end_hour == 0 else end_hour

                # Проверяем, работает ли повар в этот час
                if start_hour <= hour < end_hour:
                    chefs_online += 1

            # Курьеры
            for courier in couriers_schedule:
                start_hour = pd.to_datetime(courier['shift']['start']).hour
                end_hour = pd.to_datetime(courier['shift']['end']).hour
                end_hour = 24 if end_hour == 0 else end_hour

                if start_hour <= hour < end_hour:
                    couriers_online += 1

            # Расчет нагрузки
            orders_per_chef = orders_rate / chefs_online if chefs_online > 0 else 0
            orders_per_courier = orders_rate / couriers_online if couriers_online > 0 else 0

            # Расчет расхода сырья
            ingredients_consumption = {
                item: round(items_rate * rate, 1)
                for item, rate in items_rate_global.items()
            }

            for k in total_items_rate.keys():
                total_items_rate[k] += ingredients_consumption[k]

            res[hour] = {
                'orders_rate': orders_rate,
                'chefs_online': chefs_online,
                'couriers_online': couriers_online,
                'orders_per_chef': round(orders_per_chef, 1),
                'orders_per_courier': round(orders_per_courier, 1),
                'ingredients_consumption': ingredients_consumption
            }
        self.logger.info(f'get_forecast | current_hour: {current_hour}')
        return res, total_items_rate

    async def _run_tool_with_timeout(self, tool_name: str, func, *args, retryable: bool = True, **kwargs):
        attempts = 1 + (self.tool_retry_count if retryable else 0)

        for attempt in range(1, attempts + 1):
            started_at = time()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=self.tool_timeout_sec,
                )
                self.logger.info(
                    "Tool success | %s | attempt=%s/%s | latency=%.3fs",
                    tool_name,
                    attempt,
                    attempts,
                    time() - started_at,
                )
                return True, result
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Tool timeout | %s | attempt=%s/%s | timeout=%.1fs",
                    tool_name,
                    attempt,
                    attempts,
                    self.tool_timeout_sec,
                )
                if attempt == attempts:
                    return (
                        False,
                        f"TOOL_TIMEOUT: {tool_name} не успел выполниться за {self.tool_timeout_sec:.1f} сек",
                    )
            except Exception as e:
                self.logger.exception("Tool failed | %s", tool_name)
                return False, f"TOOL_ERROR: {tool_name} завершился с ошибкой: {e}"

    def _create_tools(self):

        @tool
        async def get_chefs_schedule(current_hour: str) -> str:
            """
            Посмотреть график поваров
            Args:
                current_hour: текущий час, в формате "Ч:00" (Например "8:00", "15:00")
            Returns:
                Список поваров
            """
            self.logger.info(f'Tool call get_chefs_schedule | current_hour: {current_hour}')

            def _impl():
                current_hour_int = pd.to_datetime(current_hour).hour
                result = _prepare_schedule_info(chefs_schedule, current_hour_int)
                return json.dumps(result, ensure_ascii=False)

            ok, result = await self._run_tool_with_timeout(
                'get_chefs_schedule',
                _impl,
            )
            return result

        @tool
        async def get_couriers_schedule(current_hour: str) -> str:
            """
            Посмотреть график курьеров
            Args:
                current_hour: текущий час, в формате "Ч:00" (Например "8:00", "15:00")
            Returns:
                Список курьеров
            """
            self.logger.info(f'Tool call get_couriers_schedule | current_hour: {current_hour}')

            def _impl():
                current_hour_int = pd.to_datetime(current_hour).hour
                result = _prepare_schedule_info(couriers_schedule, current_hour_int)
                return json.dumps(result, ensure_ascii=False)

            ok, result = await self._run_tool_with_timeout(
                'get_couriers_schedule',
                _impl,
            )
            return result

        @tool
        async def get_information_about_items(items_list: list[str]) -> str:
            """
            Проверить остатки в других пиццериях.
            Args:
                items_list: список ингредиентов в формате ['ингредиент1', 'ингредиент2']

            Returns:
                Наличие ингредиента в других пиццериях:
                    Ингридиент - название ингредиента
                    Количество - количество ингредиента в единицах измерения
                    Единица измерения - единица измерения ингредиента
                    Пиццерия - в какой пиццерии находит ингредиент
                    Можно купить в магазине - если "Да", то можно докупить в магазине. Если "Нет", то можно только взять у другой пиццерии
            """
            self.logger.info(f'Tool call get_information_about_items | items_list: {items_list}')

            def _impl():
                items = pd.read_excel('data/stores.xlsx')
                result = items[items['Ингридиент'].isin(items_list)]
                return result.to_string(index=False)

            ok, result = await self._run_tool_with_timeout(
                'get_information_about_items',
                _impl,
            )
            return result

        @tool
        async def prepare_stop_item(item_name: str) -> str:
            """
            Предложить менеджеру поставить сырье в стоп - подготовить кнопку.
            Args:
                item_name: название сырья - "тесто", "сыр", "соус", "пепперони", "грибы"
            Returns:
                Уведомление менеджера о необходимости постановки сырья в стоп
            """
            self.logger.info(f'Tool call prepare_stop_item | items_list: {item_name}')

            def _impl():
                return StopItem(item_name=item_name)

            ok, result = await self._run_tool_with_timeout(
                'prepare_stop_item',
                _impl,
                retryable=False,
            )
            if not ok:
                return result
            actions.append(result)
            return f'Кнопка для постановки сырья "{item_name}" в стоп готова'

        @tool
        async def call_employee(employee_type: str, employee_name: str, time: str) -> str:
            """
            Подготовить кнопку для вызова дополнительного сотрудника
            Args:
                employee_type: тип сотрудника chef или courier
                employee_name: имя сотрудника
                time: время, к которому нужно вызвать сотрудника
            Returns:
                Подтверждение подготовки кнопки
            """
            self.logger.info(f'Tool call call_employee | employee_type: {employee_type} | employee_name: {employee_name} | time_to: {time}')

            def _impl():
                if employee_type.lower() == 'chef':
                    return CallChef(employee_name=employee_name, time=time)
                return CallCourier(employee_name=employee_name, time=time)

            ok, result = await self._run_tool_with_timeout(
                'call_employee',
                _impl,
                retryable=False,
            )
            if not ok:
                return result
            actions.append(result)
            return f"Менеджеру будет выведена кнопка для вызова сотрудника {employee_type} ко времени {time}"

        return [get_chefs_schedule, get_couriers_schedule,
                get_information_about_items, prepare_stop_item, call_employee]

    def init_agent(self, settings: LazySettings):
        self.logger.info(f'Инициализация {self.__class__.__name__}')
        tools = self._create_tools()

        # Базовый LLM
        llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            temperature=settings.temperature,
            base_url=settings.base_url,
            reasoning_effort="high",
            timeout=120.0,
        )

        self._llm_plain = llm
        self.llm_with_tools = llm.bind_tools(tools)
        tool_node = ToolNode(tools)

        builder = StateGraph(AgentState)
        builder.add_node('agent', self.agent_node)
        builder.add_node('tools', tool_node)

        builder.set_entry_point('agent')

        builder.add_conditional_edges(
            'agent',
            self.should_continue,
            {
                'tools': 'tools',
                END: END,
            },
        )

        builder.add_edge('tools', 'agent')

        self.graph = builder.compile()
        self.logger.info('Агент готов к работе')


    def limit_messages(self, state: AgentState) -> AgentState:
        """Ограничивает историю по циклам (а не по количеству сообщений)."""
        messages = state.get('messages') or []

        # Обрезаем только на старте нового прогона, чтобы не потерять контекст внутри текущего tool-цикла.
        if state.get('tool_call_count', 0) != 0:
            return state

        max_cycles = getattr(self, "max_history_cycles", None)
        if max_cycles is None or max_cycles <= 0:
            return state

        # Система сохраняется как префикс.
        system_prefix: list = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prefix.append(msg)
            else:
                break

        # Один цикл начинается с HumanMessage (ввод метрик) и заканчивается перед следующим HumanMessage.
        human_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
        if len(human_indices) <= max_cycles:
            return state

        keep_start_index = human_indices[-max_cycles]
        state['messages'] = system_prefix + messages[keep_start_index:]
        return state

    async def agent_node(self, state: AgentState):
        state = self.limit_messages(state)
        messages = state['messages']
        tool_call_count = state.get('tool_call_count', 0)
        response = await self.llm_with_tools.ainvoke(messages)
        self._collect_token_usage(response)
        self.logger.info(response)

        tcalls = getattr(response, 'tool_calls', None) or []
        if tcalls and tool_call_count >= self.max_tool_calls:
            self.logger.warning(
                'Лимит tool_calls исчерпан; финальный ответ без инструментов.'
            )
            response = await self._llm_plain.ainvoke(messages)
            self._collect_token_usage(response)
            return {'messages': [response], 'tool_call_count': tool_call_count}

        if tcalls:
            return {'messages': [response], 'tool_call_count': tool_call_count + 1}
        return {'messages': [response], 'tool_call_count': tool_call_count}

    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        tool_call_count = state.get('tool_call_count', 0)
        tcalls = getattr(last_message, 'tool_calls', None) or []

        if tcalls:
            if tool_call_count < self.max_tool_calls:
                return 'tools'
            self.logger.warning('Лимит tool_calls достигнут; завершаем агент.')
            return END

        return END

    def clean_history(self):
        self.messages = {
            'messages': [
                SystemMessage(content=self.system_prompt),
            ],
            'tool_call_count': 0,
        }

    async def run(self, metrics: dict):
        global actions
        actions = []
        start = time()

        self.logger.info(f'Время {metrics["timestamp"]}')
        ts = pd.to_datetime(metrics["timestamp"])
        current_hour = ts.time().hour

        weather = self.weather_info[self.weather_info['Час'] >= current_hour]
        weather = weather[weather['Час'] <= current_hour + 3]
        weather_row = f'Данные о погоде \n{weather.to_string(index=False)}'

        forecast = ''
        if self.forecast_count % 4 == 0:
            forecast = self.get_forecast(current_hour)

        self.forecast_count += 1

        metrics_input = f'{weather_row}\n{json.dumps(metrics, ensure_ascii=False)}'
        if forecast:
            self.logger.info('Добавлен прогноз')
            metrics_input += f'\n{json.dumps(forecast)}'

        self.logger.info(f'Input: {metrics_input}')

        self.messages['tool_call_count'] = 0
        self.messages['messages'].append(HumanMessage(content=metrics_input))
        len_messages = len(self.messages['messages'])
        try:
            result = await self.graph.ainvoke(self.messages)
        except Exception:
            self.logger.exception("Субагент метрик: сбой выполнения графа")
            self.messages['messages'].pop()
            self.latency = time() - start
            empty = {"messages": list(self.messages["messages"]) + [AIMessage(content="")]}
            return empty, [], []

        called_tools = [used_tool.name for used_tool in result['messages'][len_messages:] if isinstance(used_tool, ToolMessage)]
        self.messages['messages'] = result['messages']

        self.latency = time() - start
        return result, actions, called_tools
