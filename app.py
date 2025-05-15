import os
import logging
import torch
from langchain.schema import Document
import streamlit as st
import numpy as np
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, login
import faiss
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity


os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"


login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])


os.makedirs('logs', exist_ok=True)


logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


valores_referencia = {
    "Hemácias (milhões/mm³)": {"H": (4.5, 6.5), "M": (4, 5)},
    "Hemoglobina (g/dL)": {"H": (13, 18), "M": (12.0, 15.5)},
    "Hematócrito (%)": {"H": (40, 54), "M": (36, 45)},
    "VCM (fL)": {"H": (80, 98), "M": (80, 98)},
    "HCM (pg)": {"H": (27, 32), "M": (27, 32)},
    "CHCM (g/dL)": {"H": (32, 36), "M": (32, 36)},
    "RDW (%)": {"H": (11, 15), "M": (11, 15)},
    "Leucócitos (/mm³)": {"H": (4000, 10000), "M": (4000, 10000)},
    "Neutrófilos Relativos (%)": {"H": (40, 75), "M": (40, 75)},
    "Neutrófilos Absolutos (/mm³)": {"H": (1600, 7500), "M": (1600, 7500)},
    "Eosinófilos Relativos (%)": {"H": (1, 5), "M": (1, 5)},
    "Eosinófilos Absolutos (/mm³)": {"H": (40, 500), "M": (40, 500)},
    "Basófilos Relativos (%)": {"H": (0, 2), "M": (0, 2)},
    "Basófilos Absolutos (/mm³)": {"H": (0, 200), "M": (0, 200)},
    "Monócitos Relativos (%)": {"H": (2, 10), "M": (2, 10)},
    "Monócitos Absolutos (/mm³)": {"H": (80, 1000), "M": (80, 1000)},
    "Linfócitos Relativos (%)": {"H": (25, 45), "M": (25, 45)},
    "Linfócitos Absolutos (/mm³)": {"H": (1000, 4500), "M": (1000, 4500)},
    "Plaquetas (/mm³)": {"H": (150000, 450000), "M": (150000, 450000)},
}


fallback_respostas = {
    "Hemácias (milhões/mm³)": {
         "baixo": "Pode indicar anemia, que pode ser causada por deficiência de ferro, vitamina B12 ou ácido fólico, hemorragia aguda ou crônica, doenças crônicas ou problemas na medula óssea. (Fonte: SBPC/ML)",
         "alto": "Sugere policitemia, que pode ser primária (como na policitemia vera) ou secundária a condições como hipóxia crônica (ex.: DPOC) ou desidratação. (Fonte: SBPC/ML)"
    },
    "Hemoglobina (g/dL)": {
         "baixo": "Indica anemia, com sintomas como cansaço, fraqueza, tontura e palidez. Causas incluem deficiências nutricionais, perda sanguínea ou doenças hematológicas. (Fonte: Ministério da Saúde do Brasil e SBPC/ML)",
         "alto": "Sugere policitemia ou desidratação. Em casos de policitemia, pode aumentar o risco de trombose. (Fonte: Ministério da Saúde do Brasil e SBPC/ML)"
    },
    "Hematócrito (%)": {
         "baixo": "Reflete anemia, com possíveis causas semelhantes às da hemoglobina baixa. (Fonte: SBPC/ML)",
         "alto": "Pode ser causado por desidratação ou policitemia, aumentando a viscosidade sanguínea e o risco de complicações cardiovasculares. (Fonte: SBPC/ML)"
    },
    "VCM (fL)": {
         "baixo": "Indica microcitose, comum em anemias ferroprivas. (Fonte: Manual de Hematologia Clínica - SBPC/ML)",
         "alto": "Sugere macrocitose, associada a deficiências de vitamina B12 ou ácido fólico. (Fonte: Manual de Hematologia Clínica - SBPC/ML)",
         "normal": "Não exclui anemia, mas ajuda a classificá-la. (Fonte: Manual de Hematologia Clínica - SBPC/ML)"
    },
    "HCM (pg)": {
         "baixo": "Sugere hipocromia, frequentemente associada à anemia ferropriva. (Fonte: SBPC/ML)",
         "alto": "Pode ocorrer em macrocitoses, como nas deficiências de vitamina B12 ou ácido fólico. (Fonte: SBPC/ML)"
    },
    "CHCM (g/dL)": {
         "baixo": "Indica hipocromia, geralmente associada à anemia ferropriva. (Fonte: SBPC/ML)",
         "alto": "Raramente alterado, mas pode ocorrer em condições como esferocitose hereditária. (Fonte: SBPC/ML)"
    },
    "RDW (%)": {
         "baixo": "Pouco relevante clinicamente. (Fonte: SBPC/ML)",
         "alto": "Sugere anisocitose, comum em anemias ferroprivas, megaloblásticas ou após tratamento de deficiências nutricionais. (Fonte: SBPC/ML)"
    },
    "Leucócitos (/mm³)": {
         "baixo": "Indica leucopenia, que pode ser causada por infecções virais, quimioterapia, doenças autoimunes ou problemas na medula óssea. (Fonte: SBPC/ML)",
         "alto": "Sugere leucocitose, comum em infecções bacterianas, inflamações, estresse ou leucemias. (Fonte: SBPC/ML)"
    },
    "Neutrófilos Relativos (%)": {
         "baixo": "Aumenta o risco de infecções, podendo ser causado por infecções virais, uso de medicamentos ou doenças hematológicas. (Fonte: SBPC/ML)",
         "alto": "Sugere infecções bacterianas, inflamações, estresse físico ou uso de corticoides. (Fonte: SBPC/ML)"
    },
    "Neutrófilos Absolutos (/mm³)": {
         "baixo": "Aumenta o risco de infecções, podendo ser causado por infecções virais, uso de medicamentos ou doenças hematológicas. (Fonte: SBPC/ML)",
         "alto": "Sugere infecções bacterianas, inflamações, estresse físico ou uso de corticoides. (Fonte: SBPC/ML)"
    },
    "Eosinófilos Relativos (%)": {
         "baixo": "Raramente clinicamente relevante. (Fonte: SBPC/ML)",
         "alto": "Associado a alergias, parasitoses, doenças autoimunes ou neoplasias hematológicas. (Fonte: SBPC/ML)"
    },
    "Eosinófilos Absolutos (/mm³)": {
         "baixo": "Raramente clinicamente relevante. (Fonte: SBPC/ML)",
         "alto": "Associado a alergias, parasitoses, doenças autoimunes ou neoplasias hematológicas. (Fonte: SBPC/ML)"
    },
    "Basófilos Relativos (%)": {
         "baixo": "Geralmente sem significado clínico. (Fonte: SBPC/ML)",
         "alto": "Pode ocorrer em doenças mieloproliferativas, como a leucemia mieloide crônica. (Fonte: SBPC/ML)"
    },
    "Basófilos Absolutos (/mm³)": {
         "baixo": "Geralmente sem significado clínico. (Fonte: SBPC/ML)",
         "alto": "Pode ocorrer em doenças mieloproliferativas, como a leucemia mieloide crônica. (Fonte: SBPC/ML)"
    },
    "Monócitos Relativos (%)": {
         "baixo": "Raramente clinicamente relevante. (Fonte: SBPC/ML)",
         "alto": "Sugere infecções crônicas, doenças inflamatórias ou neoplasias hematológicas. (Fonte: SBPC/ML)"
    },
    "Monócitos Absolutos (/mm³)": {
         "baixo": "Raramente clinicamente relevante. (Fonte: SBPC/ML)",
         "alto": "Sugere infecções crônicas, doenças inflamatórias ou neoplasias hematológicas. (Fonte: SBPC/ML)"
    },
    "Linfócitos Relativos (%)": {
         "baixo": "Pode ocorrer em infecções virais graves, imunodeficiências ou uso de medicamentos imunossupressores. (Fonte: SBPC/ML)",
         "alto": "Comum em infecções virais ou leucemias linfocíticas. (Fonte: SBPC/ML)"
    },
    "Linfócitos Absolutos (/mm³)": {
         "baixo": "Pode ocorrer em infecções virais graves, imunodeficiências ou uso de medicamentos imunossupressores. (Fonte: SBPC/ML)",
         "alto": "Comum em infecções virais ou leucemias linfocíticas. (Fonte: SBPC/ML)"
    },
    "Plaquetas (/mm³)": {
         "baixo": "Aumenta o risco de sangramento, podendo ser causado por doenças autoimunes, infecções virais, medicação ou doenças hematológicas. (Fonte: SBPC/ML)",
         "alto": "Pode ser reativa ou primária, aumentando o risco de trombose. (Fonte: SBPC/ML)"
    }
}


class CustomEmbeddings:
    def __init__(self, model_name):
        logging.info(f"Carregando modelo e tokenizer: {model_name}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)


    def embed_texts(self, texts):
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("Os textos para embeddings devem ser uma lista de strings não vazias.")


        logging.info("Calculando embeddings...")
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()


        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("Os embeddings gerados devem ser uma matriz 2D (n_texts x n_features).")


        return embeddings


    def embed_documents(self, documents):
        logging.info("Convertendo documentos para texto antes de calcular embeddings.")
        if not documents or not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("Todos os documentos devem ser instâncias de `Document`.")
        texts = [doc.page_content.strip() for doc in documents if doc.page_content.strip()]
        return self.embed_texts(texts)


def preprocessar_texto(texto):
    import re
    texto = re.sub(r"Página\s*\d+\s*de\s*\d+", "", texto)
    texto = re.sub(r"(.+)\n\1", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


@st.cache_data
def load_documents():
    try:
        index_path = '/content/drive/MyDrive/IC/faiss_index_flat.bin'
        texts_path = '/content/drive/MyDrive/IC/texts.pkl'


        if os.path.exists(index_path) and os.path.exists(texts_path):
            index = faiss.read_index(index_path)
            with open(texts_path, 'rb') as f:
                texts = pickle.load(f)
            logging.info("Índice FAISS e textos carregados do cache.")
            return index, texts


        loader = PyPDFDirectoryLoader('/content/drive/MyDrive/IC/Data/')
        raw_docs = loader.load()
        docs = [Document(page_content=preprocessar_texto(doc.page_content)) for doc in raw_docs if doc.page_content.strip()]
        logging.info(f"Documentos carregados: {len(docs)}")


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=64,
            length_function=lambda x: len(x.split())
        )
        chunks = text_splitter.split_documents(docs)
        texts = [chunk.page_content.strip() for chunk in chunks]
        logging.info(f"Chunks gerados: {len(texts)}")


        embeddings, _ = load_embeddings_and_llm()
        text_embeddings = embeddings.embed_texts(tuple(texts)).astype("float32")


        index = faiss.IndexFlatL2(text_embeddings.shape[1])
        index.add(text_embeddings)
        logging.info("Índice FAISS criado com sucesso.")


        faiss.write_index(index, index_path)
        with open(texts_path, 'wb') as f:
            pickle.dump(texts, f)


        return index, texts


    except Exception as e:
        logging.error(f"Erro ao carregar documentos: {e}")
        raise


@st.cache_resource
def load_embeddings_and_llm():
    logging.info("Carregando embeddings e LLM...")
    try:
        embeddings = CustomEmbeddings("pucpr/biobertpt-all")
        logging.info("Embeddings carregados com sucesso.")


        llm_model_path = "/content/drive/MyDrive/IC/BioMistral-7B.Q4_K_M.gguf"
        if not os.path.exists(llm_model_path):
            raise FileNotFoundError(f"Modelo LLM não encontrado: {llm_model_path}")


        llm = LlamaCpp(
            model_path=llm_model_path,
            temperature=0.2,
            max_tokens=800,
            n_ctx=8192,
            top_p=0.95,
            n_batch=512,
            verbose=False
        )


        logging.info("LLM carregado com sucesso.")
        return embeddings, llm


    except Exception as e:
        logging.error(f"Erro ao carregar embeddings ou LLM: {e}")
        st.error(f"Erro ao carregar embeddings ou LLM: {e}")
        st.stop()


def identificar_anomalias(inputs, genero):
    anormalidades = {}
    for exame, valor in inputs.items():
        if valor is not None and exame in valores_referencia:
            try:
                ref_min, ref_max = valores_referencia[exame][genero]
                if valor < ref_min:
                    diferenca = ref_min - valor
                    gravidade = classificar_gravidade(diferenca, ref_min)
                    anormalidades[exame] = (valor, "baixo", gravidade)
                elif valor > ref_max:
                    diferenca = valor - ref_max
                    gravidade = classificar_gravidade(diferenca, ref_max)
                    anormalidades[exame] = (valor, "alto", gravidade)
                else:
                    anormalidades[exame] = (valor, "normal", None)
            except KeyError:
                logging.error(f"Erro ao obter valores de referência para {exame}")
                continue
    return anormalidades


def classificar_gravidade(diferenca, referencia):
    percentual = (diferenca / referencia) * 100
    if percentual <= 10:
        return "leve"
    elif percentual <= 20:
        return "moderada"
    else:
        return "severa"


def contar_tokens(texto):
    return len(texto.split())


def filtrar_chunks_relevantes(chunks, query_embedding_np, embeddings, threshold=0.5):
    relevant_chunks = []
    for chunk in chunks:
        chunk_embedding = embeddings.embed_texts([chunk])
        similarity = cosine_similarity(query_embedding_np, chunk_embedding)[0][0]
        if similarity > threshold:
            relevant_chunks.append(chunk)
    return relevant_chunks


def formatar_chunks_para_resposta(chunks):
    """
    Formata os chunks relevantes em uma resposta legível.
    """
    resposta = "**Informações encontradas nos documentos:**\n\n"
    for i, chunk in enumerate(chunks, start=1):
        resposta += f"**Trecho {i}:**\n{chunk}\n\n"
    return resposta


def buscar_explicacao_pdf(exame, condicao, retriever, embeddings, llm):
    query = (
        f"Explicação clínica sobre {exame} estar {condicao} em exames de sangue. "
        f"Possíveis causas, significado clínico, diagnóstico diferencial, "
        f"e recomendações para {exame} {condicao}."
    )
    logging.info(f"Buscando explicação para: {query}")


    index, texts = retriever
    query_embedding = embeddings.embed_texts([query])
    query_embedding_np = np.array(query_embedding).astype("float32")


    D, I = index.search(query_embedding_np, k=2)
    similar_chunks = [texts[idx] for idx in I[0]]
    relevant_chunks = filtrar_chunks_relevantes(similar_chunks, query_embedding_np, embeddings, threshold=0.5)
    logging.info(f"Chunks relevantes encontrados: {len(relevant_chunks)}")


    def get_fallback():
        return fallback_respostas.get(exame, {}).get(
            condicao, "Explicação clínica não disponível no momento."
        )


    if not relevant_chunks:
        logging.warning("Nenhum chunk relevante encontrado. Usando fallback.")
        return get_fallback()


    contexto = "\n".join(relevant_chunks)
    logging.debug(f"Contexto para o LLM: {contexto[:500]}...")


    prompt = (
        f"Você é um assistente médico especializado em hematologia. Com base no contexto clínico abaixo, "
        f"explique de forma clara e acessível para um paciente.\n\n"
        f"**Contexto clínico relevante:**\n{contexto}\n\n"
        f"**Pergunta 1:** Explique de forma concisa o que significa {exame} {condicao} no contexto clínico.\n"
        f"**Pergunta 2:** Quais são as causas mais comuns dessa condição? Liste até 3 possíveis causas.\n"
        f"**Pergunta 3:** Quais são as ações recomendadas para acompanhamento e tratamento dessa condição? "
        f"Liste até 2 recomendações práticas.\n\n"
        f"**Instruções (Português):**\n"
        f"- Use uma linguagem simples e evite termos técnicos que o paciente não entenderia.\n"
        f"- Caso não tenha informações suficientes para uma resposta precisa, responda com 'Informação não disponível'.\n\n"
    )


    logging.debug(f"Prompt enviado ao LLM: {prompt}")


    try:
        start_time = time.time()
        resposta = llm.invoke(prompt)
        elapsed_time = time.time() - start_time
        logging.info(f"Tempo gasto na geração do LLM: {elapsed_time:.2f} segundos")
        logging.info(f"Resposta do LLM: {resposta}")


        if not resposta or "não disponível" in resposta.lower() or len(resposta.strip()) < 58:
            logging.warning("Resposta do LLM inválida ou insuficiente. Usando chunks relevantes como resposta.")
            return formatar_chunks_para_resposta(relevant_chunks)


        return resposta
    except Exception as e:
        logging.error(f"Erro ao gerar explicação com o LLM: {e}")
        return formatar_chunks_para_resposta(relevant_chunks)




def gerar_relatorio(anormalidades, retriever, llm, genero, embeddings):
    if not anormalidades:
        return "Nenhuma anomalia detectada nos exames fornecidos."


    relatorio = "# Relatório de Análise de Exames de Sangue\n\n"
    relatorio += "## Anomalias Detectadas\n\n"


    exames_normais = []


    for exame, detalhes in anormalidades.items():
        valor = detalhes[0]
        condicao = detalhes[1]
        gravidade = detalhes[2]
        ref_min, ref_max = valores_referencia.get(exame, {}).get(genero, (None, None))


        if condicao == "normal":
            exames_normais.append(exame)
            continue


        relatorio += f"### {exame}\n\n"
        relatorio += "| **Parâmetro**         | **Valor** |\n"
        relatorio += "|------------------------|-----------|\n"
        relatorio += f"| Valor Encontrado       | {valor}    |\n"
        relatorio += f"| Condição               | {condicao.capitalize()} |\n"
        relatorio += f"| Gravidade              | {gravidade.capitalize()} |\n"
        if ref_min is not None and ref_max is not None:
            relatorio += f"| Valores de Referência  | {ref_min} - {ref_max} |\n"
        relatorio += "\n"


        explicacao = buscar_explicacao_pdf(exame, condicao, retriever, embeddings, llm)
        relatorio += f"**Possível Significado Clínico:** {explicacao}\n\n"


    if exames_normais:
        relatorio += "## Exames Dentro da Faixa Normal\n\n"
        relatorio += "Os seguintes exames estão dentro dos valores de referência:\n"
        for exame in exames_normais:
            relatorio += f"- {exame}\n"
        relatorio += "\n"


    relatorio += "## Recomendações\n"
    relatorio += "- Consulte um hematologista para avaliação detalhada.\n"
    relatorio += "- Realize exames adicionais (ferritina, vitamina B12, ácido fólico, etc.).\n"
    relatorio += "- Monitore os parâmetros regularmente.\n"


    return relatorio


def main():
    embeddings, llm = load_embeddings_and_llm()
    index, texts = load_documents()


    st.title("Análise de Exames de Sangue")
    genero = st.selectbox("Selecione o Gênero", ["H", "M"])


    inputs = {
        "Hemácias (milhões/mm³)": st.number_input("Hemácias (milhões/mm³)"),
        "Hemoglobina (g/dL)": st.number_input("Hemoglobina (g/dL)"),
        "Hematócrito (%)": st.number_input("Hematócrito (%)"),
        "VCM (fL)": st.number_input("VCM (fL)"),
        "HCM (pg)": st.number_input("HCM (pg)"),
        "CHCM (g/dL)": st.number_input("CHCM (g/dL)"),
        "RDW (%)": st.number_input("RDW (%)"),
        "Leucócitos (/mm³)": st.number_input("Leucócitos (/mm³)"),
        "Neutrófilos Relativos (%)": st.number_input("Neutrófilos Relativos (%)"),
        "Neutrófilos Absolutos (/mm³)": st.number_input("Neutrófilos Absolutos (/mm³)"),
        "Eosinófilos Relativos (%)": st.number_input("Eosinófilos Relativos (%)"),
        "Eosinófilos Absolutos (/mm³)": st.number_input("Eosinófilos Absolutos (/mm³)"),
        "Basófilos Relativos (%)": st.number_input("Basófilos Relativos (%)"),
        "Basófilos Absolutos (/mm³)": st.number_input("Basófilos Absolutos (/mm³)"),
        "Monócitos Relativos (%)": st.number_input("Monócitos Relativos (%)"),
        "Monócitos Absolutos (/mm³)": st.number_input("Monócitos Absolutos (/mm³)"),
        "Linfócitos Relativos (%)": st.number_input("Linfócitos Relativos (%)"),
        "Linfócitos Absolutos (/mm³)": st.number_input("Linfócitos Absolutos (/mm³)"),
        "Plaquetas (/mm³)": st.number_input("Plaquetas (/mm³)"),
    }


    if st.button("Gerar Relatório"):
        with st.spinner("Analisando..."):
            st.write("Identificando anomalias...")
            anormalidades = identificar_anomalias(inputs, genero)


            if anormalidades:
                st.write("Gerando relatório...")
                relatorio = gerar_relatorio(anormalidades, (index, texts), llm, genero, embeddings)
                with st.expander("Relatório Completo"):
                    st.markdown(relatorio)
            else:
                st.write("Nenhuma anomalia detectada nos exames fornecidos.")


if __name__ == "__main__":
    main()



