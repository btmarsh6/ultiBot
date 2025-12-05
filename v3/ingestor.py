import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def load_documents():
    """Load and combine WFDF + USAU rule PDFs."""
    print('📚 Loading PDFs...')

    wfdf_loader = PyPDFLoader('data/WFDF-Rules-of-Ultimate-2025-2028.pdf')
    usau_loader = PyPDFLoader('data/USAU-Official-Rules-of-Ultimate-2024-2025.pdf')

    wfdf_docs = wfdf_loader.load()
    usau_docs = usau_loader.load()

    docs = wfdf_docs + usau_docs
    print(f'Loaded {len(docs)} pages total.')
    return docs


def split_documents(docs):
    '''Split documents into smaller chunks for embedding.'''
    print('✂️  Splitting documents into chunks...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(docs)
    print(f'Created {len(chunks)} chunks.')
    return chunks


def get_embeddings():
    '''Initialize BGE-large embedding model'''
    print('🔍 Initializing BGE-large embeddings model...')

    return HuggingFaceEmbeddings(
        model_name='BAAI/bge-large-en-v1.5',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def init_pinecone(index_name='ultimate-rules'):
    '''Initialize Pinecone vector store.'''
    print('🌲 Initializing Pinecone vector store...')

    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file.")

    pc = Pinecone(api_key=api_key)

    # create index if needed
    if index_name not in pc.list_indexes().names():
        print(f'Creating Pinecone index: {index_name}...')
        pc.create_index(
            name=index_name,
            dimension=1024,     # dimension of BGE-large embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
                ),
        )
    else:
        print(f'Pinecone index "{index_name}" already exists.')

    return pc.Index(index_name)


def store_embeddings(chunks, embedding_model, index_name='ultimate-rules'):
    '''Embed and store document chunks in Pinecone'''
    print('💾 Storing embeddings in Pinecone vector store...')

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name
    )

    print('✅ Embeddings stored successfully.')
    return vectorstore


if __name__ == '__main__':
    docs = load_documents()
    chunks = split_documents(docs)
    embedding_model = get_embeddings()
    index_name = 'ultimate-rules'
    index = init_pinecone(index_name)

    store_embeddings(chunks, embedding_model, index_name)

    print('Ingestion complete.')
