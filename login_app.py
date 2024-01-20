# streamlit_app.py
from sqlalchemy import create_engine, inspect, event
from typing import Dict, Any
import libsql_client
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.llms import OpenAI
import openai
import os
import pandas as pd
from llama_index.llms.palm import PaLM
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index import SQLDatabase, ServiceContext
from llama_index.indices.struct_store import NLSQLTableQueryEngine
import hmac
import streamlit as st


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

#carga de secrets
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
TURSO_DB_URL = os.environ.get("TURSO_DB_URL")
TURSO_DB_AUTH_TOKEN = os.environ.get("TURSO_DB_AUTH_TOKEN")

#creación del url
dbUrl = f"sqlite+{TURSO_DB_URL}/?authToken={TURSO_DB_AUTH_TOKEN}&secure=true"

class StreamlitChatPack(BaseLlamaPack):

    def __init__(
        self,
        page: str = "Bienvenido a tu CMMS por IA",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        
        self.page = page

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        import streamlit as st

        st.set_page_config(
            page_title=f"{self.page}",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        if "messages" not in st.session_state:  # Initialize the chat messages history
            st.session_state["messages"] = [
                {"role": "assistant", "content": f"Hola, Soy tu asistente IA del CMMS, ¿En que te apoyo?"}
            ]

        st.title(
            f"{self.page}:mechanic:"
        )
                # Sidebar Intro
        st.sidebar.markdown('### Creado Por:')
        st.sidebar.markdown("""
        JESUS FERNANDEZ: 
        [Linkedin](https://www.linkedin.com/in/jesusd-fz/)
        """)
        st.info(
            f"Guarda los datos de mantenimiento de tu empresa en el sistema.",
            icon="ℹ️",
        )

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(
                message
            )  # Add response to message history

        def get_table_data(table_name, client):
            try:
                # Ejecuta la consulta SQL y obtén el resultado con el cliente de Turso Database
                result_set = client.execute(f"SELECT * FROM {table_name}") 
                # Obtén las columnas y las filas del ResultSet
                columns = result_set.columns
                rows = result_set.rows
                # Construye un DataFrame de pandas con las columnas y filas
                df = pd.DataFrame(rows, columns=columns)
                return df

            except Exception as e:
                print(f"Error al obtener datos de la tabla '{table_name}': {e}")
                # Puedes ajustar el manejo de errores según tus necesidades
                return None  # o puedes levantar la excepción nuevamente



        @st.cache_resource
        def load_db_llm():
            # conección y carga de turso database
            engine = create_engine(dbUrl, connect_args={'check_same_thread': False}, echo=True)
            sql_database = SQLDatabase(engine) #include all tables
         

            # Initialize LLM
            #llm2 = PaLM(api_key=os.environ["GOOGLE_API_KEY"])  # Replace with your API key
            llm2 = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")

            service_context = ServiceContext.from_defaults(llm=llm2, embed_model="local")
            
            return sql_database, service_context, engine

        sql_database, service_context, engine = load_db_llm()


       # Sidebar for database schema viewer
        st.sidebar.markdown("## Visor de esquemas CMMS")

        # Create an inspector object
        inspector = inspect(engine)

        # Get list of tables in the database
        table_names = inspector.get_table_names()

        # Sidebar selection for tables
        selected_table = st.sidebar.selectbox("La tabla seleccionada es:", table_names)

        client = libsql_client.create_client_sync(url=TURSO_DB_URL, auth_token=TURSO_DB_AUTH_TOKEN)
        
        # Display the selected table
        if selected_table:
            df = get_table_data(selected_table, client)
            st.sidebar.text(f"datos para la tabla: '{selected_table}':")
            st.sidebar.dataframe(df)
    

        if "query_engine" not in st.session_state:  # Initialize the query engine
            st.session_state["query_engine"] = NLSQLTableQueryEngine(
                sql_database=sql_database,
                synthesize_response=True,
                service_context=service_context
            )

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input(
            "Ingresa tu petición en lenguaje natural"
        ):  # Prompt for user input and save to chat history
            with st.chat_message("user"):
                st.write(prompt)
            add_to_message_history("user", prompt)

        # If last message is not from assistant, generate a new response
        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.spinner():
                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query("User Question:"+prompt+". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)                   
                    
                    
                    st.sidebar.markdown("## Resultados de la Consulta SQL")
                    
                    try:
                        with engine.connect() as conn, conn.begin():
                            df_result = pd.read_sql_query(response.metadata['sql_query'],conn)
                            st.sidebar.dataframe(df_result)
                    except Exception as e:
                        st.sidebar.error(f"Error al ejecutar la consulta: {e}")
                    finally:
                        conn.close()


                    add_to_message_history("assistant", sql_query)


if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
