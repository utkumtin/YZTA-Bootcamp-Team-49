# Gizlilik

Bu belge, uygulamanın verinizle ne yaptığını anlatır. İki soruyu ayrı ayrı yanıtlar:
hangi garantiler kodun kendisi tarafından zorlanıyor, hangi noktalarda üçüncü tarafın
taahhüdüne güveniyoruz. Zorlanamayan bir şey garanti gibi sunulmaz.

## İki mod

Gizlilik modu her sayfada sol panelden seçilir ve tüm model çağrılarını yönlendirir.

| | Herkese açık | Özel |
|---|---|---|
| Kullanım amacı | Kamuya açık veri, demo, jüri gösterimi | Yayımlanmamış veya hassas veri |
| Seçilebilen uçlar | Ücretsiz katman dahil tüm küratörlü uçlar | Yalnız eğitim yapmayan uçlar |
| Sağlayıcı eğitimi | Mümkün, ücretsiz katmanda beklenmelidir | Sağlayıcı taahhüdüyle kapalı |
| Anahtar bulunamazsa | Zincirdeki diğer sağlayıcıya geçilir | Açık hata verilir, ücretsiz uca düşülmez |

Özel modda yönlendirme iki kez kontrol edilir: sağlayıcı listesi zaten yalnız eğitim
yapmayan uçları içerir, ayrıca zincir kurulduktan sonra her üyenin bu niteliği taşıdığı
çalışma anında doğrulanır. Doğrulama başarısız olursa istek kurulmaz, uygulama hata
yükseltir. Bu davranış test altındadır.

Ücretsiz ve ücretli Gemini uçları ayrı anahtar değişkenleri kullanır. Ücretsiz anahtarın
ücretli slota devredilmesi mümkün değildir: elinizde yalnız ücretsiz anahtar varken özel
modda çalıştırırsanız uygulama sessizce ücretsiz uca geçmez, eksik anahtarı bildirir.

## Modele ne gidiyor

Yüklediğiniz dosyanın ham satırları hiçbir modda modele gönderilmez. Dosyadan modele giden
yük, deterministik olarak üretilen kolon düzeyinde bir özettir:

- satır ve kolon sayısı, yinelenen satır sayısı, birleştirme anahtarı adayları
- her kolon için veri tipi, eksik değer sayısı ve oranı, benzersiz değer sayısı
- sayısal kolonlarda en küçük, en büyük, ortalama ve standart sapma
- kategorik kolonlarda en sık görülen beş değer ve frekansları

Bu özetin iki yerinde gerçek veri değerleri görünür: kategorik kolonların en sık beş
değeri ve sayısal kolonların en küçük ile en büyük değerleri. Dolayısıyla iddia "hiçbir
veri değeri dışarı çıkmaz" değil, "ham satırlar dışarı çıkmaz" ve "bir satırın alanları
bir arada dışarı çıkmaz" biçimindedir. Kişiyi tanımlayabilecek mikro veri yüklemeyin.

Kolon adları ve örnek değerler kullanıcı verisinden geldiği için modele gönderilmeden
önce güvenilmez olarak işaretlenir. Bu işaretleme, veriye gömülü bir metnin talimat gibi
okunma olasılığını azaltır; tek başına yeterli bir savunma değildir.

Yukarıdaki özet yalnız temizleme adımını anlatır. Sonraki adımlar modele veri dosyasından
değil, sizin yazdıklarınızdan ve deterministik hesap çıktılarından beslenir: estimand
kurulurken yazdığınız araştırma hikayesi ile Sokratik beyan metni, menü kurulurken
dondurulmuş estimand ve kolon adları listesi, varyans anlatısı yazılırken çokluevren
sonuç özeti ve eksen teşhisi. Bunların hiçbiri satır düzeyinde veri taşımaz; ancak serbest
metin alanlarına kendiniz hassas bir şey yazarsanız o metin modele olduğu gibi gider.

## Anahtarlar

Uygulama hiçbir API anahtarını depoya, çalışma dizinine veya çıktı paketine yazmaz.
Anahtar üç kaynaktan çözülür: oturum içinde girdiğiniz anahtar, ortam değişkeni, dağıtım
sırları. Oturum içinde girilen anahtar yalnız o oturumun belleğinde tutulur ve sekme
kapandığında kaybolur. Ayarlar sekmesindeki anahtar durumu göstergesi anahtarın kendisini
değil, yalnız hangi kaynaktan geldiğini gösterir.

## Yerel çıktılar

Karar defteri, üretilen temizleme kodu, dondurulmuş estimand ve menü kayıtları, çokluevren
sonuçları ve şekiller yerel çalışma dizinine yazılır. Bu çıktılar hiçbir uzak servise
gönderilmez, telemetri toplanmaz.

Model yanıtları, tekrar koşularını hızlandırmak ve ücretsiz katman istek limitini korumak
için yerel diske önbelleklenir. Önbellek dosyasının adı isteğin özetinden türetilen bir
karmadır, yani gönderilen istem düz metin olarak saklanmaz; yanıt gövdesi ise düz JSON
olarak yazılır ve içinde kolon adları ile karar gerekçeleri bulunabilir.
`PARETO_LLM_CACHE=0` ortam değişkeni önbelleği her iki modda da tamamen kapatır.

Özel modda önbellek public moddan **ayrışır**: yanıtlar `runs/llm_cache` yerine
`runs/llm_cache_private/<oturum>` altına yazılır ve oturum bitince silinir. Silme üç
yoldan yapılır: özel moddan herkese açık moda dönüldüğünde o oturumun dizini hemen
silinir; tarayıcısı kapatılarak terk edilmiş oturumların dizinleri bir saatlik süre
sonunda süpürülür; uygulama normal şekilde kapandığında o sürecin yazdığı dizinler
kaldırılır.

**Sınır — bunlar garanti değil, en iyi çabadır.** Streamlit'in "oturum bitti" için genel
bir kancası yoktur, bu yüzden "sekme kapandığı anda silinir" denemez. Uygulama kapanış
temizliği de yalnız düzgün kapanışta çalışır; süreç `SIGKILL` ile ya da makine kapanarak
sonlanırsa dizinler bir sonraki açılışta TTL süpürmesiyle gider. Bu aralıkta özel veriden
türeyen yanıtlar diskte kalır. Kesinlik isteyen bir oturumda tek kesin yol
`PARETO_LLM_CACHE=0` ile önbelleği baştan kapatmaktır.

## Sağlayıcı taahhütleri

| Uç | Taahhüt | Hangi modda |
|---|---|---|
| Gemini ücretsiz katman | Gönderilen içerik model geliştirmede kullanılabilir | Yalnız herkese açık |
| Gemini ücretli katman | Eğitimde kullanılmama taahhüdü ve veri işleme sözleşmesi | Özel |
| Groq | Hesap düzeyinde veri saklamama ayarı | Her iki mod |
| OpenRouter | İstek düzeyinde sıfır veri saklama zorlaması | Özel modda zorlanır |

Özel modda mekanik çağrılar tek bir uca, Groq'a sabittir ve bu seçim kullanıcıya açılmaz.
Anahtarı ise ayrı değildir: yargı için Groq seçip kendi anahtarınızı kaydettiyseniz mekanik
çağrılar da aynı anahtarla, yani sizin hesabınız üzerinden gider. Veri saklamama ayarının
açık olması gereken hesap da o hesaptır. Anahtarı siz vermediyseniz dağıtımın ortam
değişkenindeki anahtar kullanılır. Yargı çağrılarında sağlayıcıyı siz seçersiniz, ancak
seçim yalnız yukarıdaki eğitim yapmayan uçlar arasından yapılabilir.

OpenRouter tarafında ücretsiz uçlar veri saklama konusunda taahhüt vermediği için özel
modun seçenek listesinde yer almaz.

## Neyi garanti etmiyoruz

- Sağlayıcının kendi taahhüdüne uyduğunu doğrulayamayız. Zorladığımız şey, isteğin yalnız
  taahhüt veren uca kurulmasıdır.
- Groq tarafındaki veri saklamama ayarı bir hesap ayarıdır. Uygulama bunu ne okuyabilir ne
  de açabilir. Özel modda Groq seçtiğinizde uygulama bu sorumluluğu size hatırlatır.
- İstem enjeksiyonuna karşı tam bağışıklık iddia etmiyoruz. Katmanlı savunma, kararların
  kapalı bir sözlükten seçilmesi ve insan onayı, olası bir enjeksiyonun etki alanını
  daraltır; sıfıra indirmez.
- Kişisel veri mevzuatına tam uyum iddiası yoktur. Ücretli katman ve veri işleme sözleşmesi
  bunun yolunu açar, ancak bu sürüm bir uyum ürünü değildir.

## Testlerle zorlanan iddialar

Aşağıdaki maddelerin her biri otomatik testlerle doğrulanır ve sürekli entegrasyonda
koşar:

- Özel modda hiçbir rol için eğitim yapmayan olmayan bir uç seçilemez.
- Ortam değişkeniyle model kimliği değiştirilse bile bu garanti bozulmaz; sağlayıcı,
  anahtar değişkeni ve eğitim niteliği kod tarafında sabittir.
- Elde yalnız ücretsiz Gemini anahtarı varken özel mod isteği kurmaz, açık hata verir.
- Özel modda model nesnesi kurulurken yalnız eğitim yapmayan uçların anahtar değişkenleri
  okunur.
- Eğitim niteliği doğrulaması dekoratif değildir: kontrol listesine eğitim yapan bir uç
  yerleştirildiğinde çağrı hata yükseltir.
- OpenRouter özel ucunda sıfır veri saklama zorlaması istek ayarlarına kadar taşınır.
- Özel modun seçenek listelerinde ücretsiz uç kimliği bulunmaz.
- Temizleme adımında modele giden yük, en sık beş değer ile en küçük/en büyük dışında
  hiçbir hücre değeri taşımaz.
- Uygulama içindeki gizlilik notu iki modda aynı metni göstermez: özel modda zorlamayı,
  herkese açık modda eğitim uyarısını söyler.
