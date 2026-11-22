"""Show a rating prediction or unrated movie recommendations, without refitting."""
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', type=int, default=1, help='Original MovieLens user ID (1-943)')
    parser.add_argument('--item', type=int, help='Original MovieLens item ID (1-1682)')
    parser.add_argument('--top', type=int, default=5)
    args = parser.parse_args()
    with np.load(ROOT/'results/model_item_selected.npz') as archive:
        model = {key:archive[key] for key in archive.files}
    if not 1 <= args.user <= len(model['user_mean']):
        parser.error('User must be between 1 and 943')
    if args.item is not None and not 1 <= args.item <= len(model['item_mean']):
        parser.error('Item must be between 1 and 1682')
    if args.top < 1:
        parser.error('--top must be positive')
    user = args.user-1
    titles={}
    for line in (ROOT/'data/ml-100k/u.item').read_text(encoding='latin-1').splitlines():
        fields=line.split('|'); titles[int(fields[0])-1]=fields[1]
    records=np.loadtxt(ROOT/'data/ml-100k/u1.base',dtype=int)
    rated=set(records[records[:,0]==args.user,1]-1)

    def predict(item):
        if model['item_count'][item]==0:
            return float(model['user_mean'][user])
        if model['user_count'][user]==0:
            return float(model['item_mean'][item])
        return float(np.clip(model['user_mean'][user] + model['P'][user] @ model['Q'][item],1,5))

    print(f"User {args.user}; item-mean SVD; k={int(model['k'])}; rating range 1-5")
    if args.item is not None:
        item=args.item-1
        print(f"{titles[item]}: {predict(item):.3f} stars")
        print('Already rated in the development data.' if item in rated else 'Unrated in the development data.')
    else:
        candidates=[(item,predict(item)) for item in range(len(model['item_mean'])) if item not in rated]
        candidates.sort(key=lambda pair:(-pair[1],pair[0]))
        for rank,(item,score) in enumerate(candidates[:args.top],1):
            print(f'{rank}. Item {item+1}: {titles[item]} ({score:.3f})')


if __name__=='__main__':
    main()
