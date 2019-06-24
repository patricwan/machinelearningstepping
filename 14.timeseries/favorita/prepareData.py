from Utils import *


df_2019, promo_2019, items, stores = load_data()

save_unstack(df_2019, promo_2019, "pw")

