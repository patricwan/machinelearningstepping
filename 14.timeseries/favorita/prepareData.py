from Utils import *


df_2017, promo_2017, items, stores = load_data()

save_unstack(df_2017, promo_2017, "pw")

