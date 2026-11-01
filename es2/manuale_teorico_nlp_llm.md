# Computational Linguistics e Natural Language Processing

## Manuale teorico introduttivo

## Introduzione: che cosa studia il Natural Language Processing?

Il **Natural Language Processing (NLP)** è il campo dell'intelligenza
artificiale che studia come i computer possano elaborare il linguaggio
umano, comprenderne le strutture e utilizzarlo per svolgere attività
come classificazione, traduzione, ricerca di informazioni e generazione
di testo.

Un testo scritto contiene diversi livelli di informazione:

-   **Livello morfologico:** come sono costruite le parole.
-   **Livello lessicale:** quali parole e significati sono presenti.
-   **Livello sintattico:** come le parole si organizzano in una frase.
-   **Livello semantico:** quale significato esprimono le parole e le
    frasi.

L'NLP sviluppa tecniche per rappresentare e analizzare questi livelli
attraverso il calcolo.

Alcuni metodi si basano su regole linguistiche esplicite; altri
utilizzano algoritmi di Machine Learning che apprendono dai dati.

Un **Large Language Model (LLM)**, come quelli basati sull'architettura
Transformer, utilizza reti neurali profonde per modellare il linguaggio
e generare sequenze testuali. I concetti introdotti in questo manuale
costituiscono alcune delle basi teoriche dell'NLP, anche se gli LLM
moderni non devono necessariamente utilizzare tutti questi strumenti
come componenti separate.

------------------------------------------------------------------------

# 1. Morfologia

## 1.1 Che cos'è la morfologia?

La morfologia è la branca della linguistica che studia la struttura
interna delle parole e i processi attraverso i quali le parole vengono
formate o modificate.

Consideriamo la parola inglese `cats`. Possiamo scomporla in `cat` e
`-s`: la prima parte indica il concetto di gatto, mentre il suffisso
normalmente indica il plurale.

La morfologia è importante nell'NLP perché parole diverse possono
rappresentare forme differenti della stessa unità lessicale.

## 1.2 Morfemi, radici e affissi

Un **morfema** è la più piccola unità linguistica dotata di significato
o di una funzione grammaticale.

Gli affissi sono elementi che vengono aggiunti a una base lessicale:

-   **Prefissi:** precedono la base, come `un-` in `unhappy`.
-   **Suffissi:** seguono la base, come `-ness` in `happiness`.

Una parola può contenere più morfemi. Per esempio, `unhappiness` può
essere analizzata come `un + happy + ness`.

La **radice** è l'elemento lessicale fondamentale da cui si possono
formare altre parole. Non coincide necessariamente con il risultato
prodotto da uno stemmer.

## 1.3 Inflessione e derivazione

L'inflessione e la derivazione sono due processi morfologici differenti.

### Inflessione

L'inflessione modifica una parola per esprimere informazioni
grammaticali, senza creare normalmente una nuova unità lessicale.

  Forma di base   Forma flessa   Informazione
  --------------- -------------- -------------------------
  cat             cats           Plurale
  walk            walked         Passato
  walk            walking        Forma verbale in -ing
  play            plays          Terza persona singolare

Le forme flesse possono generalmente essere ricondotte allo stesso
lemma.

### Derivazione

La derivazione forma nuove parole mediante l'aggiunta di affissi o altri
processi morfologici.

  Parola di partenza   Parola derivata
  -------------------- -----------------
  happy                happiness
  happy                unhappy
  teach                teacher
  kind                 kindness

La derivazione può modificare il significato e la categoria
grammaticale.

**Distinzione fondamentale:** l'inflessione produce forme grammaticali
di una parola; la derivazione contribuisce alla formazione di nuove
parole lessicali.

## 1.4 Lemma, stem e normalizzazione

Quando analizziamo un testo, possiamo voler ricondurre forme diverse a
una rappresentazione comune. Questa operazione viene generalmente
chiamata normalizzazione lessicale.

### Stemming

Lo **stemming** è una tecnica che riduce una parola attraverso regole di
trasformazione, spesso eliminando suffissi.

Per esempio, uno stemmer potrebbe ricondurre `studies` a `studi`.

La forma ottenuta è detta *stem*. Lo stem non deve necessariamente
essere una parola esistente o una forma grammaticalmente corretta.

Lo stemming è generalmente rapido e utile quando interessa ridurre le
differenze ortografiche tra parole, per esempio nei sistemi di ricerca.

Uno stemmer come il Porter Stemmer utilizza regole organizzate per
trasformare le forme delle parole. Non cerca necessariamente il loro
lemma attraverso un dizionario.

### Lemmatizzazione

La **lemmatizzazione** cerca di ricondurre una forma al suo **lemma**,
cioè alla forma lessicale di riferimento.

Esempi:

-   `dogs → dog`
-   `children → child`
-   `running → run`, se interpretato come verbo
-   `better → good`, in determinate interpretazioni grammaticali

La lemmatizzazione può utilizzare informazioni lessicali e grammaticali
per individuare la forma di riferimento.

La categoria grammaticale può essere determinante. Per esempio,
`meeting` può essere un sostantivo oppure una forma verbale.
L'interpretazione influenza il lemma appropriato.

### Differenze tra stemming e lemmatizzazione

  -----------------------------------------------------------------------
  Aspetto                 Stemming                Lemmatizzazione
  ----------------------- ----------------------- -----------------------
  Obiettivo               Ridurre le forme a una  Ricondurre una forma al
                          radice approssimativa   lemma

  Metodo                  Regole di               Informazioni lessicali
                          trasformazione          e grammaticali

  Risultato               No                      Normalmente sì
  necessariamente valido                          
  come parola?                                    

  Utilizzo del contesto   Generalmente limitato   Possibile, secondo lo
  grammaticale                                    strumento
  -----------------------------------------------------------------------

La lemmatizzazione non è automaticamente migliore in qualsiasi
situazione. La scelta dipende dal problema da risolvere.

### Il caso di NLTK

NLTK è una libreria Python che mette a disposizione strumenti per
l'elaborazione del linguaggio naturale. Il suo `WordNetLemmatizer`
utilizza la risorsa lessicale WordNet.

Un aspetto importante è che il metodo non determina automaticamente, in
ogni contesto, la categoria grammaticale corretta di una parola. Per
impostazione predefinita, considera il sostantivo.

Per questo, forme come `studies` possono richiedere l'indicazione
esplicita della categoria verbale per ottenere il lemma `study`.

La lemmatizzazione dipende quindi anche dalle informazioni che vengono
fornite allo strumento.

------------------------------------------------------------------------

# 2. Regular expressions e tokenizzazione

## 2.1 Che cos'è la tokenizzazione?

La **tokenizzazione** è il processo mediante il quale un testo viene
suddiviso in unità chiamate **token**. Un token può essere una parola,
un numero, un segno di punteggiatura oppure una parte di parola.

Consideriamo:

`Hello, world!`

Una possibile tokenizzazione è:

`["Hello", ",", "world", "!"]`

Un altro sistema potrebbe ignorare la punteggiatura e restituire
soltanto `["Hello", "world"]`.

Non esiste una sola strategia valida per ogni situazione. La scelta
dipende dall'obiettivo dell'analisi e dal tipo di testo.

### Perché è importante?

Le decisioni prese durante la tokenizzazione influenzano tutte le
elaborazioni successive. Un sistema deve stabilire come trattare parole
con apostrofi, numeri decimali, URL, segni di punteggiatura, parole
composte e forme linguistiche rare.

Se un URL viene diviso in più parti, un sistema di ricerca potrebbe non
trattarlo più come un unico indirizzo. Se un numero decimale viene
suddiviso in tre token, la rappresentazione del testo cambia.

La tokenizzazione è quindi una fase fondamentale dell'elaborazione
testuale.

## 2.2 Che cos'è una regular expression?

Una **regular expression**, o **regex**, è un pattern che descrive una
famiglia di stringhe.

In Python, il modulo `re` permette di cercare e riconoscere sequenze di
caratteri attraverso questi pattern.

Per esempio, `r"\d+"` riconosce una sequenza di una o più cifre. In una
stringa come `I have 3 cats and 12 dogs`, il pattern individua `3` e
`12`.

### Simboli fondamentali

  -----------------------------------------------------------------------
  Simbolo                             Significato
  ----------------------------------- -----------------------------------
  `\d`                                Una cifra numerica

  `\w`                                Un carattere alfanumerico o
                                      underscore secondo le regole del
                                      motore regex

  `\s`                                Un carattere di spaziatura

  `+`                                 Una o più ripetizioni

  `*`                                 Zero o più ripetizioni

  `?`                                 Elemento facoltativo, in questo
                                      contesto

  `.`                                 Qualsiasi carattere, normalmente
                                      escluso il newline

  `\.`                                Un punto letterale

  `[abc]`                             Uno dei caratteri indicati

  `(...)`                             Un gruppo di cattura

  `(?:...)`                           Un gruppo non catturante

  `\|`                                Alternativa tra pattern
  -----------------------------------------------------------------------

Nelle espressioni regolari, la barra verticale `|` separa le
alternative. In una regex Python, normalmente si scrive semplicemente
`|`, senza la barra inversa.

Per esempio, `cat|dog` corrisponde a `cat` oppure a `dog`.

### Quantificatori

Il quantificatore `+` richiede almeno un'occorrenza. Quindi `\d+`
riconosce `7`, `42` e `2026`.

Il quantificatore `?` rende facoltativo l'elemento precedente. Per
esempio, `\$?\d+` può riconoscere sia `19` sia `$19`.

Le parentesi quadre definiscono insiemi di caratteri. Per esempio,
`[.,!?]` riconosce un singolo carattere tra punto, virgola, punto
esclamativo e punto interrogativo.

## 2.3 Tokenizzazione basata su regex

Un tokenizer basato su regex definisce esplicitamente quali sequenze
riconoscere. Un pattern può contenere alternative per URL, numeri,
parole e punteggiatura.

L'ordine delle alternative può influenzare il risultato, perché un
pattern generale può intercettare una parte della sequenza prima che
venga riconosciuta da un pattern più specifico.

Per questo, quando si vogliono preservare intere strutture come gli URL,
è importante considerare le regole di riconoscimento e l'ordine delle
alternative.

Le regex offrono controllo e flessibilità, ma richiedono che il
progettista stabilisca esplicitamente i casi da riconoscere.

## 2.4 NLTK e spaCy

NLTK e spaCy sono due librerie utilizzate nell'NLP.

**NLTK** offre una raccolta di strumenti per la ricerca e l'analisi
linguistica, tra cui tokenizer, corpora e risorse lessicali.

**spaCy** offre una pipeline NLP integrata, che comprende la
tokenizzazione e, a seconda del modello utilizzato, l'analisi
grammaticale, il parsing e altre elaborazioni.

Un tokenizer basato su regex segue i pattern definiti dall'utente. Un
tokenizer linguistico può invece incorporare regole specifiche per una
lingua e gestire convenzioni linguistiche più articolate.

La differenza fondamentale non è che un metodo sia sempre corretto e
l'altro sbagliato: ciascuno produce una segmentazione in base alle
proprie regole.

------------------------------------------------------------------------

# 3. Sintassi e Part-of-Speech Tagging

## 3.1 Che cos'è la sintassi?

La sintassi studia come le parole si combinano per formare frasi e quali
relazioni grammaticali esistono tra loro.

Consideriamo `The cat eats fish.` Possiamo identificare `cat` come
soggetto, `eats` come verbo e `fish` come oggetto.

La sintassi non si limita a riconoscere le parole. Cerca di descrivere
la loro funzione e la loro organizzazione nella frase.

## 3.2 Che cos'è il POS tagging?

POS significa *Part of Speech*, cioè categoria grammaticale. Il **POS
tagging** consiste nell'assegnare una categoria grammaticale a ogni
token.

  Categoria   Significato                Esempi
  ----------- -------------------------- -----------------
  NOUN        Sostantivo                 cat, house
  VERB        Verbo                      eat, run
  ADJ         Aggettivo                  happy, large
  ADV         Avverbio                   quickly, slowly
  PRON        Pronome                    she, they
  DET         Determinante               the, a
  ADP         Adposizione                in, on, with
  NUM         Numero                     three, 10
  CCONJ       Congiunzione coordinante   and, but
  PUNCT       Punteggiatura              , . !

Le etichette esatte dipendono dal sistema di annotazione. Nel notebook
compare il sistema Universal POS, che utilizza un insieme standardizzato
di categorie grammaticali.

### Perché il contesto è importante?

Una stessa parola può assumere categorie grammaticali differenti.

`I book a flight.` --- Qui `book` è un verbo.

`I read a book.` --- Qui `book` è un sostantivo.

Il POS tagging deve quindi considerare la parola nel contesto, non
soltanto la sua forma isolata.

## 3.3 Corpus linguistico e annotazione

Un **corpus** è una raccolta organizzata di testi utilizzabile per
analisi linguistiche e computazionali. Un corpus annotato contiene
informazioni aggiuntive associate ai testi.

Per esempio, a ogni parola possono essere assegnate una categoria
grammaticale, una relazione sintattica o altre etichette linguistiche.

Queste annotazioni possono fungere da esempi di riferimento per
addestrare e valutare modelli automatici.

Il Penn Treebank è un corpus ampiamente utilizzato nella ricerca
linguistica computazionale, che contiene testi annotati con informazioni
sintattiche e grammaticali.

## 3.4 Come un modello di Machine Learning esegue il POS tagging

Il Machine Learning permette a un algoritmo di apprendere regolarità dai
dati. In un classificatore POS, il compito consiste nell'associare a un
token una categoria grammaticale.

Un sistema tradizionale può seguire questo processo:

1.  Riceve un token e informazioni sul suo contesto.
2.  Estrae caratteristiche utili.
3.  Converte le caratteristiche in numeri.
4.  Utilizza un modello per produrre una previsione.
5.  Confronta le previsioni con le etichette corrette durante
    l'addestramento.

### Features

Le **features**, o caratteristiche, sono informazioni utilizzate da un
modello per prendere una decisione. Per un token possono comprendere la
parola stessa, le lettere iniziali o finali, la parola precedente, la
parola successiva, la presenza di numeri o di un trattino.

Queste informazioni possono fornire indizi sulla categoria grammaticale.
Per esempio, il suffisso `-ly` è spesso associato agli avverbi inglesi,
ma non costituisce una regola assoluta.

Il modello apprende come combinare gli indizi sulla base dei dati
disponibili.

### Rappresentazione numerica

Gli algoritmi di Machine Learning operano su rappresentazioni numeriche.

Una rappresentazione testuale può essere convertita in un vettore di
caratteristiche. Un metodo come `CountVectorizer` crea una
rappresentazione basata sulle caratteristiche testuali osservate.

In una rappresentazione binaria, il valore 1 può indicare che una
caratteristica è presente, mentre 0 indica che è assente.

Queste rappresentazioni possono essere sparse: la maggior parte delle
caratteristiche possibili può non essere presente in un singolo esempio.

### Rete neurale MLP

Una MLP, o *Multilayer Perceptron*, è una rete neurale costituita da più
livelli di unità computazionali.

Nel caso del POS tagging, il modello riceve un vettore numerico e
produce punteggi associati alle possibili categorie grammaticali. La
categoria con il punteggio più alto può essere scelta come previsione.

### Addestramento

Durante l'addestramento, il modello riceve esempi e le relative
etichette corrette.

La **loss**, o funzione di perdita, misura quanto le previsioni
differiscono dalle risposte attese secondo un criterio matematico.

L'**optimizer** utilizza informazioni sui gradienti per aggiornare i
parametri del modello.

La backpropagation permette di calcolare i gradienti necessari
all'aggiornamento dei parametri.

Un'**epoca** corrisponde normalmente a un passaggio completo attraverso
i dati di addestramento.

Ripetendo questo processo, il modello può apprendere regolarità utili
per classificare nuovi esempi.

## 3.5 Baseline e accuratezza

Una baseline è un metodo semplice utilizzato come riferimento per
valutare un sistema più complesso. Per esempio, un classificatore
elementare potrebbe assegnare sempre la categoria più frequente nel
corpus.

Il modello addestrato dovrebbe essere confrontato con questo riferimento
per verificare se riesce effettivamente a migliorare il risultato.

L'accuratezza è la proporzione di previsioni corrette:

**Accuracy = previsioni corrette / previsioni totali**

Se 80 previsioni su 100 sono corrette, l'accuratezza è pari all'80%.

L'accuratezza è utile, ma non descrive tutti gli aspetti delle
prestazioni: per valutazioni più approfondite si utilizzano anche
precisione, richiamo e F1-score.

------------------------------------------------------------------------

# 4. Dependency Parsing

## 4.1 Che cos'è il dependency parsing?

Il **dependency parsing** è un'attività NLP che identifica le relazioni
sintattiche tra le parole di una frase.

Mentre il POS tagging assegna categorie grammaticali, il dependency
parsing cerca di stabilire quali parole dipendano da altre.

Consideriamo `Students read books.` Il verbo `read` costituisce il
centro della frase. `Students` è il soggetto del verbo e `books` è
l'oggetto del verbo.

Queste relazioni possono essere rappresentate attraverso un albero di
dipendenze.

## 4.2 Head, dependent e dependency relation

In una struttura di dipendenze, ogni token è associato a una testa
sintattica (*head*) e a una relazione grammaticale.

La parola che dipende da un'altra è detta *dependent*.

Le relazioni sintattiche possono essere indicate da etichette
specifiche, come:

-   `nsubj`: soggetto nominale;
-   `obj`: oggetto;
-   `amod`: modificatore aggettivale;
-   `det`: determinante;
-   `ROOT`: radice della struttura sintattica, secondo il sistema
    utilizzato.

Le etichette precise possono variare tra formalismi e strumenti.

## 4.3 Che cos'è un albero di dipendenze?

Un albero di dipendenze è una struttura composta da nodi e archi. I nodi
rappresentano i token; gli archi rappresentano le relazioni tra teste e
dipendenti.

Un albero di dipendenze ben formato deve rispettare alcune proprietà.

-   **Connessione:** tutti i token appartengono alla stessa struttura.
-   **Assenza di cicli:** non devono esistere percorsi diretti che
    permettano di tornare a un nodo di partenza seguendo gli archi.
-   **Singola testa:** ogni token della frase possiede una sola testa
    sintattica.
-   **Proiettività:** in un albero proiettivo, gli archi possono essere
    rappresentati senza incroci nella sequenza lineare delle parole,
    rispettando la definizione formale di proiettività.

La proiettività è una proprietà più restrittiva delle altre. Una
struttura può essere un albero valido e tuttavia essere non proiettiva.

## 4.4 Ambiguità sintattica

Una frase è sintatticamente ambigua quando ammette più strutture
grammaticali plausibili.

Consideriamo `I saw the man with the telescope.` La frase può
significare che ho usato un telescopio per vedere l'uomo, oppure che ho
visto un uomo che aveva un telescopio.

Le parole sono le stesse, ma le relazioni sintattiche possono essere
differenti.

Un parser deve selezionare un'analisi sulla base del proprio modello e
delle informazioni disponibili. La struttura prodotta non garantisce
necessariamente che tutte le altre interpretazioni siano impossibili.

## 4.5 Parsing basato su transizioni

Un parser *transition-based* costruisce progressivamente un albero
attraverso una sequenza di operazioni.

Nella variante Arc-standard, la configurazione contiene tre elementi:

-   **STACK:** la pila degli elementi attualmente in elaborazione.
-   **BUFFER:** le parole ancora da elaborare.
-   **ARCS:** gli archi sintattici già costruiti.

L'algoritmo utilizza tre transizioni principali.

**SHIFT:** sposta la prima parola del buffer nello stack.

**LEFT-ARC:** crea una dipendenza tra gli elementi in cima allo stack,
orientando l'arco dalla cima verso l'elemento immediatamente
sottostante. Il dipendente viene quindi rimosso dallo stack.

**RIGHT-ARC:** crea una dipendenza dall'elemento immediatamente
sottostante verso quello in cima allo stack. Il dipendente viene rimosso
dallo stack.

Queste operazioni consentono di costruire l'albero gradualmente. La
variante Arc-standard standard è limitata agli alberi proiettivi.

### Static oracle

Un *oracle* è un meccanismo che indica quale transizione eseguire in una
determinata configurazione.

Uno static oracle conosce l'albero corretto di riferimento e sceglie le
azioni compatibili con la sua costruzione. Questo è diverso da un parser
addestrato che deve prevedere le transizioni senza conoscere in anticipo
l'albero corretto della frase.

------------------------------------------------------------------------

# 5. Semantica e ambiguità lessicale

## 5.1 Che cos'è la semantica?

La semantica studia il significato delle parole, delle espressioni e
delle frasi.

La semantica lessicale si concentra in particolare sui significati delle
parole e sulle relazioni tra i concetti che esse esprimono.

La parola inglese `bank`, per esempio, può indicare un istituto
finanziario oppure la riva di un fiume. La forma è identica, ma il
significato dipende dal contesto.

## 5.2 Polisemia e omonimia

La **polisemia** si verifica quando una parola possiede significati
distinti ma correlati. Per esempio, `paper` può indicare il materiale
cartaceo oppure un articolo accademico.

L'**omonimia** riguarda invece forme identiche associate a significati
distinti e non correlati, o non sufficientemente correlati da essere
trattati come manifestazioni dello stesso senso lessicale.

Un esempio è `bark`, che può indicare l'abbaiare di un cane oppure la
corteccia di un albero.

La distinzione può richiedere analisi lessicografiche più approfondite,
ma entrambi i fenomeni mostrano che una forma ortografica non determina
sempre un unico significato.

## 5.3 Contesto e significato

Il contesto permette di individuare indizi utili per interpretare una
parola ambigua.

`I deposited money at the bank.`

Il riferimento al denaro favorisce il significato finanziario.

`We sat on the bank of the river.`

Il riferimento al fiume favorisce il significato di riva.

Il significato non dipende quindi soltanto dalla parola isolata, ma
anche dalle altre parole e dalla situazione comunicativa.

------------------------------------------------------------------------

# 6. WordNet e le relazioni lessicali

## 6.1 Che cos'è WordNet?

WordNet è una banca dati lessicale dell'inglese che organizza le parole
secondo i loro significati e le relazioni semantiche tra i concetti.

Non è un modello generativo: è una risorsa strutturata che può essere
interrogata mediante strumenti software.

WordNet è utilizzato nell'NLP per esplorare significati, relazioni
lessicali e gerarchie concettuali.

## 6.2 Synset e lemma

Un **synset** è un insieme di lemmi che condividono uno specifico
significato.

Un lemma è una forma lessicale di riferimento.

Una stessa parola può essere associata a più synset, perché può
possedere diversi sensi.

Un synset contiene normalmente: - Un identificatore. - Una categoria
grammaticale. - Una definizione, detta *gloss*. - Eventuali esempi. -
Relazioni con altri synset.

Un identificatore come `car.n.01` distingue uno specifico senso nominale
di `car`. Il numero identifica il senso all'interno dell'organizzazione
della risorsa; non rappresenta un punteggio di qualità.

## 6.3 Relazioni semantiche

### Sinonimia

Due lemmi possono condividere uno stesso senso. Per esempio, `car` e
`automobile` possono essere sinonimi in determinati contesti.

La sinonimia non implica che due parole siano intercambiabili in
qualsiasi frase.

### Iperonimia

L'iperonimia esprime una relazione tra un concetto specifico e uno più
generale.

`Car → vehicle`

Una macchina è un tipo di veicolo. Quindi `vehicle` è un iperonimo di
`car`.

### Iponimia

L'iponimia esprime la relazione inversa.

`Car` è un iponimo di `vehicle`, perché rappresenta una categoria più
specifica.

Gli iperonimi e gli iponimi formano spesso gerarchie concettuali.

### Antonimia

L'antonimia collega parole che esprimono significati opposti in un
determinato senso.

Esempi: `hot` e `cold`; `happy` e `unhappy`.

Le relazioni dipendono dai sensi considerati e non si applicano
necessariamente a ogni uso di una parola.

## 6.4 Relazioni dirette e indirette

Una relazione diretta collega immediatamente due synset. Una relazione
indiretta richiede invece di attraversare uno o più collegamenti.

Se `car` è un tipo di `vehicle` e `vehicle` è un tipo di `artifact`,
`artifact` è un concetto più generale rispetto a `car`, ma non è
necessariamente il suo iperonimo diretto.

Questa distinzione è importante quando si consultano le relazioni
fornite da WordNet.

## 6.5 Similarità semantica

La similarità semantica cerca di misurare quanto due concetti siano
vicini secondo una determinata rappresentazione.

Una misura basata sui percorsi in WordNet considera la distanza tra due
synset nella gerarchia semantica. In generale, due concetti collegati da
percorsi relativamente brevi possono essere considerati più simili
secondo questa metrica.

Tuttavia, la similarità ottenuta dipende dai sensi selezionati, dalla
struttura della risorsa e dalla metrica utilizzata.

Una misura di similarità non equivale a una comprensione completa del
significato.

------------------------------------------------------------------------

# 7. Word Sense Disambiguation

## 7.1 Definizione

La **Word Sense Disambiguation (WSD)** è il compito di selezionare il
significato appropriato di una parola in un determinato contesto.

Se una parola possiede più sensi, il sistema deve individuare quello che
meglio si adatta alla frase.

Per esempio, `The fisherman sat on the bank` suggerisce il significato
di riva del fiume.

La WSD cerca di formalizzare questo processo attraverso algoritmi
linguistici o modelli statistici e neurali.

## 7.2 L'algoritmo di Lesk

L'algoritmo di Lesk è un metodo classico per la disambiguazione del
significato delle parole.

L'idea generale consiste nel confrontare le parole del contesto con le
definizioni dei sensi candidati.

Per ciascun senso: 1. Si recupera la definizione. 2. Si confrontano le
parole della definizione con quelle del contesto. 3. Si calcola un
punteggio di sovrapposizione. 4. Si seleziona il senso con il punteggio
migliore.

Il principio è che una definizione che condivide più parole con il
contesto potrebbe rappresentare meglio il significato utilizzato nella
frase.

### Simplified Lesk

La versione semplificata confronta direttamente il contesto con la
definizione del senso candidato.

Se una definizione contiene parole presenti anche nella frase, può
ottenere un punteggio maggiore.

Il metodo è semplice, ma dipende dalla sovrapposizione letterale tra le
parole. Se il contesto contiene `river` e la definizione usa un sinonimo
come `watercourse`, un confronto letterale potrebbe non riconoscere la
relazione semantica.

### Original Lesk

La versione originale utilizza un confronto più ampio tra le definizioni
dei sensi candidati e quelle associate alle parole del contesto.

In questo modo può individuare sovrapposizioni anche tra definizioni
correlate, anziché limitarsi alle parole presenti direttamente nella
frase.

La versione originale può quindi sfruttare informazioni semantiche
aggiuntive, anche se il risultato dipende dalle definizioni disponibili.

### Limiti di Lesk

Lesk presenta alcuni limiti: - Le definizioni possono essere brevi o
incomplete. - La stessa idea può essere espressa con parole
differenti. - Le forme flesse possono ridurre le corrispondenze
letterali. - Un punteggio elevato non garantisce che il senso
selezionato sia corretto. - Più sensi possono ottenere punteggi uguali.

È un metodo utile per comprendere la disambiguazione, ma non risolve
perfettamente ogni ambiguità.

## 7.3 Similarità coseno

Un'altra strategia consiste nel rappresentare il contesto e le
definizioni mediante vettori numerici.

La similarità coseno confronta la direzione dei due vettori:

\[ `\operatorname{cos}`{=tex}(A,B)=`\frac{A\cdot B}{\|A\|\|B\|}`{=tex}
\]

Dove: - (A) e (B) sono vettori numerici; - (A`\cdot `{=tex}B) è il
prodotto scalare; - (\|A\|) e (\|B\|) sono le rispettive norme.

Se i vettori rappresentano frequenze di parole, una similarità elevata
indica una maggiore somiglianza nella distribuzione delle
caratteristiche rappresentate.

Se le componenti sono non negative, i valori sono normalmente compresi
tra 0 e 1.

La similarità coseno può essere utilizzata per confrontare testi, ma il
risultato dipende da come vengono costruiti i vettori. Due testi possono
avere una similarità elevata senza esprimere esattamente lo stesso
significato.

------------------------------------------------------------------------

# 8. Il collegamento con gli LLM

Gli argomenti precedenti appartengono all'NLP, ma non tutti gli LLM
utilizzano esplicitamente gli strumenti tradizionali descritti.

Gli LLM moderni si basano generalmente su reti neurali profonde e,
frequentemente, sull'architettura Transformer. Per comprenderne il
funzionamento è utile conoscere alcuni concetti introduttivi.

## Tokenizzazione nei modelli linguistici

Un LLM non riceve direttamente le parole come concetti astratti. Il
testo viene suddiviso in token e trasformato in identificatori numerici.

A differenza dei tokenizer linguistici tradizionali, i tokenizer degli
LLM spesso utilizzano unità *subword*: una parola può essere
rappresentata da un token oppure suddivisa in più parti.

## Rappresentazioni numeriche

I token vengono associati a rappresentazioni numeriche, spesso chiamate
**embedding**.

Gli embedding sono vettori che permettono al modello di elaborare le
informazioni linguistiche mediante operazioni matematiche.

## Apprendimento dai dati

Durante l'addestramento, il modello modifica i propri parametri sulla
base di grandi quantità di dati.

Molti LLM apprendono a prevedere token successivi o sequenze di token,
anche se le procedure complessive di addestramento possono comprendere
ulteriori obiettivi e fasi.

## Differenza rispetto agli strumenti tradizionali

Gli strumenti tradizionali possono utilizzare regole esplicite,
dizionari, relazioni lessicali codificate e modelli addestrati per
compiti specifici.

Un LLM può invece apprendere rappresentazioni linguistiche attraverso
l'addestramento su grandi quantità di testo.

Questo non significa che gli LLM siano sempre più precisi in ogni
attività, né che non possano utilizzare risorse esterne. Significa che
il loro approccio alla modellazione del linguaggio è differente.

------------------------------------------------------------------------

# Glossario essenziale

  Termine              Definizione
  -------------------- ------------------------------------------------------------
  NLP                  Elaborazione computazionale del linguaggio umano
  Morfologia           Studio della struttura interna delle parole
  Morfema              Unità minima dotata di significato o funzione grammaticale
  Lemma                Forma lessicale di riferimento
  Stem                 Forma ridotta ottenuta attraverso un processo di stemming
  Stemming             Riduzione delle parole mediante regole di trasformazione
  Lemmatizzazione      Ricondurre una forma al suo lemma
  Token                Unità individuata durante la tokenizzazione
  Tokenizzazione       Suddivisione di un testo in token
  Regex                Pattern per riconoscere sequenze di caratteri
  POS                  Categoria grammaticale
  Corpus               Raccolta organizzata di testi
  Feature              Caratteristica utilizzata da un modello
  MLP                  Rete neurale multilivello
  Loss                 Misura dell'errore di un modello
  Optimizer            Algoritmo per aggiornare i parametri
  Baseline             Metodo di riferimento per valutare un sistema
  Dependency parsing   Analisi delle relazioni sintattiche
  Head                 Testa sintattica di un token
  Dependent            Token che dipende da una testa
  ROOT                 Radice della struttura sintattica
  Synset               Insieme di lemmi che condividono un senso
  Iperonimo            Concetto più generale
  Iponimo              Concetto più specifico
  WSD                  Disambiguazione del significato delle parole
  Embedding            Rappresentazione numerica di un elemento linguistico
  LLM                  Modello linguistico di grandi dimensioni

## Conclusione

I concetti di questo manuale descrivono diversi modi di rappresentare il
linguaggio: dalla struttura delle parole alle relazioni grammaticali,
fino ai significati lessicali.

La distinzione fondamentale è tra **forma, funzione grammaticale,
struttura sintattica e significato**.

Una parola può avere una determinata forma, appartenere a una categoria
grammaticale, svolgere una funzione sintattica e assumere un significato
specifico nel contesto. Queste informazioni sono correlate, ma non sono
intercambiabili.

Comprendere questa distinzione costituisce la base teorica necessaria
per leggere con autonomia il codice e le spiegazioni di un'introduzione
alla linguistica computazionale.
